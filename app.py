from pathlib import Path
from datetime import datetime, timezone
import math

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml-model"

# All supported prediction models.
# TCS uses the existing .joblib model; the other stocks use
# the newly trained percentage-return .pkl bundles.
MODEL_FILES = {
    "TCS.NS": "stock_prediction_model.joblib",
    "RELIANCE.NS": "RELIANCE_NS_RandomForest.pkl",
    "HDFCBANK.NS": "HDFCBANK_NS_RandomForest.pkl",
    "ICICIBANK.NS": "ICICIBANK_NS_RandomForest.pkl",
    "INFY.NS": "INFY_NS_RandomForest.pkl",
    "SBIN.NS": "SBIN_NS_RandomForest.pkl",
    "ITC.NS": "ITC_NS_RandomForest.pkl",
    "BHARTIARTL.NS": "BHARTIARTL_NS_RandomForest.pkl",
    "WIPRO.NS": "WIPRO_NS_RandomForest.pkl",
    "TATASTEEL.NS": "TATASTEEL_NS_RandomForest.pkl",
}

MODELS = {}

for stock_symbol, filename in MODEL_FILES.items():
    model_path = MODEL_DIR / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required model file not found: {model_path}\n"
            "Check that all 10 model files are inside the ml-model folder."
        )

    try:
        MODELS[stock_symbol] = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model for {stock_symbol} from {model_path}: {exc}"
        ) from exc


app = FastAPI(
    title="StockVision AI - Stock Prediction API",
    version="2.0.0",
    description="Market data and experimental stock prediction for 10 NSE stocks."
)


# ============================================================
# SYMBOL VALIDATION AND COMMON HELPERS
# ============================================================

def validate_indian_symbol(symbol: str) -> str:
    """Allow Indian NSE (.NS) and BSE (.BO) symbols."""
    symbol = symbol.strip().upper()

    if (
        not symbol
        or len(symbol) > 30
        or not symbol.endswith((".NS", ".BO"))
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Indian stocks only. Use an NSE ticker ending in .NS "
                "or BSE ticker ending in .BO."
            )
        )

    return symbol


def clean_number(value):
    """Convert a value to a finite Python float, or None."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance versions that return MultiIndex columns."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def download_prediction_history(symbol: str) -> pd.DataFrame:
    """Download enough daily history to calculate all model features."""
    try:
        data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance download failed: {str(exc)[:200]}"
        ) from exc

    if data is None or data.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No market data received from Yahoo Finance for {symbol}."
        )

    data = flatten_yfinance_columns(data)
    data = data.dropna(subset=["Close"]).copy()

    required_columns = {"Close", "High", "Low", "Volume"}
    missing = sorted(required_columns.difference(data.columns))

    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Market data for {symbol} is missing columns: {missing}"
        )

    return data


# ============================================================
# FEATURE ENGINEERING - EXISTING TCS MODEL
# ============================================================

def calculate_tcs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve the feature definitions used by the existing TCS model.
    Do not change these formulas unless the TCS model is retrained.
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["Return1"] = close.pct_change(1)
    df["Return5"] = close.pct_change(5)

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()

    df["Volatility10"] = df["Return1"].rolling(10).std()
    df["Range"] = (high - low) / close
    df["VolumeChange"] = volume.pct_change()

    df["Momentum3"] = close.pct_change(3)
    df["Momentum10"] = close.pct_change(10)
    df["Momentum20"] = close.pct_change(20)
    df["Momentum60"] = close.pct_change(60)

    df["DistanceMA10"] = close / df["MA10"] - 1
    df["DistanceMA20"] = close / df["MA20"] - 1
    df["DistanceMA50"] = close / close.rolling(50).mean() - 1

    df["Volatility20"] = df["Return1"].rolling(20).std()
    df["IntradayRange"] = (high - low) / close

    return df


# ============================================================
# FEATURE ENGINEERING - THE OTHER 9 RANDOM FOREST MODELS
# ============================================================

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    These formulas match the feature engineering used when the
    nine new Random Forest models were trained in Colab.
    """
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    features["Return_1D"] = close.pct_change(1) * 100
    features["Return_2D"] = close.pct_change(2) * 100
    features["Return_5D"] = close.pct_change(5) * 100
    features["Return_10D"] = close.pct_change(10) * 100

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    features["MACD_Normalized"] = macd / close
    features["MACD_Signal_Normalized"] = macd_signal / close

    return features


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Stock Prediction API is running",
        "total_models": len(MODELS),
        "supported_stocks": list(MODELS.keys()),
        "model_directory": str(MODEL_DIR)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loaded_models": len(MODELS),
        "supported_stocks": list(MODELS.keys())
    }


# ============================================================
# MULTI-STOCK AI PREDICTION
# ============================================================

@app.get("/predict")
def predict(symbol: str = Query(default="TCS.NS")):
    symbol = validate_indian_symbol(symbol)

    if symbol not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No prediction model is configured for {symbol}. "
                f"Supported symbols: {', '.join(MODELS.keys())}"
            )
        )

    try:
        bundle = MODELS[symbol]
        data = download_prediction_history(symbol)

        # TCS has its original feature set and target convention.
        if symbol == "TCS.NS":
            feature_data = calculate_tcs_features(data)
        else:
            feature_data = calculate_momentum_features(data)

        # Feature names and order are read from the saved model bundle.
        model_features = bundle.get("features")

        if not model_features:
            raise ValueError(
                f"The saved model bundle for {symbol} has no 'features' list."
            )

        missing_features = [
            feature for feature in model_features
            if feature not in feature_data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Feature engineering for {symbol} did not produce: "
                f"{missing_features}"
            )

        valid_rows = feature_data.dropna(subset=model_features)

        if valid_rows.empty:
            raise HTTPException(
                status_code=503,
                detail=f"Not enough valid market history to predict {symbol}."
            )

        latest_row = valid_rows.iloc[[-1]][model_features]

        if not np.isfinite(latest_row.to_numpy(dtype=float)).all():
            raise ValueError("Latest feature row contains NaN or infinite values.")

        estimator = bundle.get("model")
        if estimator is None:
            raise ValueError(f"Model estimator missing from bundle for {symbol}.")

        raw_prediction = float(estimator.predict(latest_row)[0])

        latest_date = latest_row.index[-1]
        latest_close = float(data.loc[latest_date, "Close"])

        if not math.isfinite(latest_close) or latest_close <= 0:
            raise ValueError("Latest closing price is invalid.")

        # TCS's existing model predicts a fractional return, e.g. 0.00412.
        # The nine new models predict percentage points, e.g. 0.412.
        if symbol == "TCS.NS":
            predicted_return_percent = raw_prediction * 100
        else:
            predicted_return_percent = raw_prediction

        predicted_close = latest_close * (
            1 + predicted_return_percent / 100
        )

        if not math.isfinite(predicted_close):
            raise ValueError("Predicted closing price is invalid.")

        # New bundles store evaluation metrics inside test_results.
        # The existing TCS bundle stores mae/rmse at its top level.
        test_results = bundle.get("test_results") or {}

        mae = bundle.get("mae", test_results.get("MAE"))
        rmse = bundle.get("rmse", test_results.get("RMSE"))

        model_type = (
            bundle.get("model_type")
            or bundle.get("model_name")
            or "Random Forest"
        )

        return {
            "stock_symbol": symbol,
            "last_trading_date": str(latest_date.date()),
            "latest_close": round(latest_close, 2),
            "predicted_next_close": round(predicted_close, 2),
            "predicted_return_percent": round(predicted_return_percent, 3),
            "model_type": model_type,
            "mae": clean_number(mae),
            "rmse": clean_number(rmse),
            "currency": "INR",
            "data_type": "latest_available_daily_close",
            "note": (
                "Experimental model estimate based on the latest available "
                "daily market data. It is not a guaranteed future price "
                "or financial advice."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for {symbol}: {str(exc)[:250]}"
        ) from exc


# ============================================================
# SHARED MARKET DATA
# ============================================================

def get_stock(symbol: str):
    symbol = validate_indian_symbol(symbol)

    try:
        stock = yf.Ticker(symbol)
        history = stock.history(
            period="1y",
            interval="1d",
            auto_adjust=False
        )

        if history.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {symbol}. Check the ticker."
            )

        return stock, history

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Market data provider failed: {str(exc)[:180]}"
        ) from exc


# ============================================================
# MARKET QUOTE
# ============================================================

@app.get("/market/quote")
def market_quote(symbol: str = Query(default="TCS.NS")):
    symbol = validate_indian_symbol(symbol)
    stock, history = get_stock(symbol)

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest

    close = clean_number(latest.get("Close"))
    previous_close = clean_number(previous.get("Close"))

    change = (
        close - previous_close
        if close is not None and previous_close is not None
        else None
    )

    change_percent = (
        (change / previous_close) * 100
        if change is not None and previous_close not in (None, 0)
        else None
    )

    try:
        info = stock.get_info()
    except Exception:
        info = {}

    return {
        "symbol": symbol,
        "currency": "INR",
        "latest_date": str(history.index[-1].date()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "latest_available_historical_quote",
        "close": close,
        "previous_close": previous_close,
        "change": clean_number(change),
        "change_percent": clean_number(change_percent),
        "open": clean_number(latest.get("Open")),
        "high": clean_number(latest.get("High")),
        "low": clean_number(latest.get("Low")),
        "volume": clean_number(latest.get("Volume")),
        "fifty_two_week_high": clean_number(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": clean_number(info.get("fiftyTwoWeekLow")),
        "market_cap": clean_number(info.get("marketCap")),
        "provider": "Yahoo Finance via yfinance",
        "note": "Latest available data; not guaranteed to be real-time."
    }


# ============================================================
# HISTORICAL PRICES
# ============================================================

@app.get("/market/history")
def market_history(
    symbol: str = Query(default="TCS.NS"),
    period: str = Query(default="1mo"),
    interval: str = Query(default="1d")
):
    symbol = validate_indian_symbol(symbol)

    allowed_periods = {
        "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"
    }
    allowed_intervals = {"1d", "1wk", "1mo"}

    if period not in allowed_periods:
        raise HTTPException(status_code=400, detail="Unsupported period")

    if interval not in allowed_intervals:
        raise HTTPException(status_code=400, detail="Unsupported interval")

    try:
        history = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False
        )

        if history.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No history found for {symbol}"
            )

        rows = []

        for date, row in history.iterrows():
            rows.append({
                "date": str(date.date()),
                "open": clean_number(row.get("Open")),
                "high": clean_number(row.get("High")),
                "low": clean_number(row.get("Low")),
                "close": clean_number(row.get("Close")),
                "volume": clean_number(row.get("Volume"))
            })

        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "count": len(rows),
            "prices": rows
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to fetch history: {str(exc)[:180]}"
        ) from exc


# ============================================================
# COMPANY FUNDAMENTALS
# ============================================================

@app.get("/market/company")
def market_company(symbol: str = Query(default="TCS.NS")):
    symbol = validate_indian_symbol(symbol)
    stock, history = get_stock(symbol)

    try:
        info = stock.get_info()
    except Exception:
        info = {}

    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("exchange"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "business_summary": info.get("longBusinessSummary"),
        "market_cap": clean_number(info.get("marketCap")),
        "trailing_pe": clean_number(info.get("trailingPE")),
        "forward_pe": clean_number(info.get("forwardPE")),
        "eps": clean_number(info.get("trailingEps")),
        "dividend_yield": clean_number(info.get("dividendYield")),
        "currency": "INR",
        "fifty_two_week_high": clean_number(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": clean_number(info.get("fiftyTwoWeekLow")),
        "latest_available_date": str(history.index[-1].date())
    }


# ============================================================
# DYNAMIC STOCK SEARCH
# ============================================================

@app.get("/market/search")
def market_search(query: str = Query(min_length=1, max_length=80)):
    """
    Search company names and ticker symbols through yfinance.
    Results depend on Yahoo Finance coverage.
    """
    search_text = query.strip()

    if not search_text:
        raise HTTPException(
            status_code=400,
            detail="Enter a company name or ticker"
        )

    try:
        search = yf.Search(
            search_text,
            max_results=10,
            news_count=0
        )
        quotes = getattr(search, "quotes", []) or []

        results = []
        seen = set()

        for item in quotes:
            symbol = item.get("symbol")

            if not symbol or symbol in seen:
                continue

            symbol = symbol.upper()

            # Indian exchange listings only.
            if not symbol.endswith((".NS", ".BO")):
                continue

            quote_type = (item.get("quoteType") or "").upper()

            if quote_type and quote_type not in {
                "EQUITY", "ETF", "MUTUALFUND", "INDEX"
            }:
                continue

            seen.add(symbol)

            results.append({
                "symbol": symbol,
                "name": (
                    item.get("longname")
                    or item.get("shortname")
                    or item.get("displayName")
                    or symbol
                ),
                "exchange": item.get("exchDisp") or item.get("exchange"),
                "type": quote_type or None
            })

        return {
            "query": search_text,
            "count": len(results),
            "results": results,
            "provider": "Yahoo Finance via yfinance",
            "note": (
                "Indian NSE/BSE listings only. "
                "Results depend on Yahoo Finance coverage."
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Stock search provider failed: {str(exc)[:180]}"
        ) from exc
