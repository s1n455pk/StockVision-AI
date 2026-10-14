
package com.stockmarket.stock_prediction;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

@RestController
@RequestMapping("/api/market")
public class MarketController {

    private final RestClient restClient = RestClient.create();

    private static final String PYTHON_API =
            "http://127.0.0.1:8000";

    @GetMapping("/quote")
    public String quote(
            @RequestParam(defaultValue = "TCS.NS") String symbol) {

        return restClient.get()
                .uri(PYTHON_API + "/market/quote?symbol={symbol}", symbol)
                .retrieve()
                .body(String.class);
    }

    @GetMapping("/history")
    public String history(
            @RequestParam(defaultValue = "TCS.NS") String symbol,
            @RequestParam(defaultValue = "1mo") String period,
            @RequestParam(defaultValue = "1d") String interval) {

        return restClient.get()
                .uri(PYTHON_API
                                + "/market/history?symbol={symbol}&period={period}&interval={interval}",
                        symbol, period, interval)
                .retrieve()
                .body(String.class);
    }

    @GetMapping("/company")
    public String company(
            @RequestParam(defaultValue = "TCS.NS") String symbol) {

        return restClient.get()
                .uri(PYTHON_API + "/market/company?symbol={symbol}", symbol)
                .retrieve()
                .body(String.class);
    }

    @GetMapping("/search")
    public String search(
            @RequestParam String query) {

        return restClient.get()
                .uri(PYTHON_API + "/market/search?query={query}", query)
                .retrieve()
                .body(String.class);
    }
}
