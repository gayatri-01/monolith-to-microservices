package com.monolith.poc.controller;

import com.monolith.poc.model.Order;
import com.monolith.poc.service.OrderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/orders")
public class OrderController {

    private final OrderService orderService;

    @Autowired
    public OrderController(OrderService orderService) {
        this.orderService = orderService;
    }

    @PostMapping("/place")
    public Order placeOrder(@RequestParam Long userId, @RequestParam Long productId) {
        return orderService.placeOrder(userId, productId);
    }

    @GetMapping
    public List<Order> getOrders() {
        return orderService.getAllOrders();
    }

    @PostMapping("/refund")
    public void processPayment(@RequestParam Long orderId, @RequestParam double amount) {
         orderService.processRefund(Long.toString(orderId), amount);
    }
}