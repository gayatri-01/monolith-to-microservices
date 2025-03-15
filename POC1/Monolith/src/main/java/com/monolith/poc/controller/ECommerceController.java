package com.monolith.poc.controller;

import com.monolith.poc.model.*;
import com.monolith.poc.service.ECommerceService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/ecommerce")
public class ECommerceController {
    private final ECommerceService ecommerceService;

    public ECommerceController(ECommerceService ecommerceService) {
        this.ecommerceService = ecommerceService;
    }

    @PostMapping("/users")
    public User createUser(@RequestBody User user) {
        return ecommerceService.createUser(user);
    }

    @GetMapping("/users")
    public List<User> getUsers() {
        return ecommerceService.getAllUsers();
    }

    @PostMapping("/products")
    public Product createProduct(@RequestBody Product product) {
        return ecommerceService.createProduct(product);
    }

    @GetMapping("/products")
    public List<Product> getProducts() {
        return ecommerceService.getAllProducts();
    }

    @PostMapping("/orders")
    public Order placeOrder(@RequestParam Long userId, @RequestParam Long productId) {
        return ecommerceService.placeOrder(userId, productId);
    }

    @GetMapping("/orders")
    public List<Order> getOrders() {
        return ecommerceService.getAllOrders();
    }

    @PostMapping("/payments")
    public Payment processPayment(@RequestParam Long orderId) {
        return ecommerceService.processPayment(orderId);
    }
}

