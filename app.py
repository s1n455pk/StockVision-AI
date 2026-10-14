from pathlib import Path
from datetime import datetime, timezone
import math

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query

# ---------- MODEL SETUP ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml-model" / "stock_prediction_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
TRAINED_SYMBOL = bundle["stock_symbol"]

app = FastAPI(title="Stock Prediction API")


# ---------- FEATURE ENGINEERING ----------
def calculate_features(df):
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


@app.get("/")
def home():
    return {
        "message": "Stock Prediction API is running",
        "trained_symbol": TRAINED_SYMBOL
    }


# ---------- AI PREDICTION ----------
@app.get("/predict")
def predict(symbol: str = Query(default="TCS.NS")):
    symbol = validate_indian_symbol(symbol)

    # IMPORTANT: current saved model was trained for one ticker only.
    if symbol != TRAINED_SYMBOL.upper():
        raise HTTPException(
            status_code=400,
            detail=(
                f"The current AI model is trained for {TRAINED_SYMBOL} only. "
                "Market data search can support other symbols, but predictions "
                "for them need a model trained and evaluated for those stocks."
            )
        )

    try:
        data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            raise HTTPException(
                status_code=503,
                detail="No market data received from Yahoo Finance"
            )

        # Handle yfinance versions that return MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = ["Close", "High", "Low", "Volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing market columns: {missing}"
            )

        features_df = calculate_features(data)
        latest = features_df.dropna(subset=FEATURES)

        if latest.empty:
            raise HTTPException(
                status_code=503,
                detail="Not enough valid market data to calculate features"
            )

        row = latest.iloc[[-1]][FEATURES]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=500,
                detail="Latest feature row contains invalid values"
            )

        predicted_return = float(model.predict(row)[0])
        latest_close = float(latest["Close"].iloc[-1])
        predicted_close = latest_close * (1 + predicted_return)

        return {
            "stock_symbol": symbol,
            "last_trading_date": str(latest.index[-1].date()),
            "latest_close": round(latest_close, 2),
            "predicted_next_close": round(predicted_close, 2),
            "predicted_return_percent": round(predicted_return * 100, 3),
            "model_type": bundle["model_type"],
            "mae": bundle["mae"],
            "rmse": bundle["rmse"],
            "note": "Experimental estimate, not a guaranteed price."
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Prediction failed: {str(exc)[:180]}"
        ) from exc


# ---------- SHARED MARKET DATA HELPERS ----------
def clean_number(value):
    """Convert missing or invalid values to JSON-safe numbers."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def validate_indian_symbol(symbol: str) -> str:
    """Allow NSE (.NS) and BSE (.BO) tickers only."""
    symbol = symbol.strip().upper()
    if not symbol or len(symbol) > 30 or not symbol.endswith((".NS", ".BO")):
        raise HTTPException(
            status_code=400,
            detail="Indian stocks only. Use an NSE ticker ending in .NS or BSE ticker ending in .BO."
        )
    return symbol


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


# ---------- MARKET QUOTE ----------
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
        if change is not None and previous_close
        else None
    )

    info = {}
    try:
        info = stock.get_info()
    except Exception:
        pass

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


# ---------- HISTORICAL PRICES ----------
@app.get("/market/history")
def market_history(
    symbol: str = Query(default="TCS.NS"),
    period: str = Query(default="1mo"),
    interval: str = Query(default="1d")
):
    symbol = validate_indian_symbol(symbol)
    allowed_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}
    allowed_intervals = {"1d", "1wk", "1mo"}

    if period not in allowed_periods:
        raise HTTPException(status_code=400, detail="Unsupported period")
    if interval not in allowed_intervals:
        raise HTTPException(status_code=400, detail="Unsupported interval")

    try:
        history = yf.Ticker(symbol.strip().upper()).history(
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
            "symbol": symbol.strip().upper(),
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


# ---------- COMPANY FUNDAMENTALS ----------
@app.get("/market/company")
def market_company(symbol: str = Query(default="TCS.NS")):
    stock, history = get_stock(symbol)
    try:
        info = stock.get_info()
    except Exception:
        info = {}

    return {
        "symbol": symbol.strip().upper(),
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


# ---------- DYNAMIC STOCK SEARCH ----------
@app.get("/market/search")
def market_search(query: str = Query(min_length=1, max_length=80)):
    """
    Search company names and ticker symbols dynamically through yfinance.
    Results depend on Yahoo Finance coverage.
    """
    search_text = query.strip()
    if not search_text:
        raise HTTPException(
            status_code=400,
            detail="Enter a company name or ticker"
        )

    try:
        search = yf.Search(search_text, max_results=10, news_count=0)
        quotes = getattr(search, "quotes", []) or []

        results = []
        seen = set()

        for item in quotes:
            symbol = item.get("symbol")
            if not symbol or symbol in seen:
                continue

            symbol = symbol.upper()
            # Indian exchange listings only: NSE (.NS) or BSE (.BO).
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
                "Indian NSE/BSE listings only. Results depend on Yahoo Finance coverage."
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Stock search provider failed: {str(exc)[:180]}"
        ) from exc