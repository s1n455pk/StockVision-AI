package com.stockmarket.stock_prediction;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

@RestController
public class StockController {

    private final RestClient restClient = RestClient.create();

    @GetMapping("/api/stock/predict")
    public String predict(
            @RequestParam(defaultValue = "TCS.NS") String symbol) {

        return restClient.get()
                .uri("http://127.0.0.1:8000/predict?symbol={symbol}", symbol)
                .retrieve()
                .body(String.class);
    }
}