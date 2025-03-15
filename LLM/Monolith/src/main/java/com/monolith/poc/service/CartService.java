package com.monolith.poc.service;

import com.monolith.poc.exeption.UnauthorizedException;
import com.monolith.poc.model.Cart;
import com.monolith.poc.model.CartItem;
import com.monolith.poc.model.Order;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;

@Service
public class CartService {

    PaymentGateway paymentGateway;



    public Cart getCartByUserId(Long userId) {
        // Implementation here
        return null;
    }

    public Cart removeItemFromCart(Long userId, Long itemId) {
        // Implementation here
        return null;
    
    }


     /**
     * Processes checkout by calculating the total and calling PaymentGateway.
     */
    public Order checkout(Long userId, List<CartItem> items) {
        double total = items.stream().map(item -> item.getPrice()).collect(Collectors.summingDouble(Double::doubleValue));
        // Call PaymentGateway to process payment
        boolean paymentSuccess = paymentGateway.processPayment(Long.toString(userId), total);
        if (paymentSuccess) {
            return new Order(Long.toString(userId), items);
        } else {
            throw new RuntimeException("Payment failed");
        }
    }

    // Verify JWT token and check for customer role
    private boolean verifyCustomerToken(String token) {
        try {          
            Claims claims = Jwts.parserBuilder()
            .build()
            .parseClaimsJws(token)
            .getBody();
            
            String role = claims.get("role", String.class);
            return "customer".equals(role);
        } catch (Exception e) {
            return false; // Invalid signature
        }
    }

    // Add item to cart, requiring customer authentication
    public void addItemToCart(Long userId, CartItem cartItem) {
        
        if (!verifyCustomerToken(getTokenFromHeader())) {
            throw new UnauthorizedException("Invalid or missing customer token");
        }
        // Logic to add item to cart
        //.....
    }


    
    // Dummy method to get token from header
    private String getTokenFromHeader() {
        // Dummy implementation to get token from header
        return "dummyToken";
    }
}