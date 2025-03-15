package com.monolith.poc.service;

import java.util.List;

import com.monolith.poc.model.CartItem;
import com.monolith.poc.model.Order;

public class OrderService {
    
 PaymentGateway paymentGateway;
    
 public Order createOrder(Long userId, List<CartItem> cartItems) {
  // TODO implement here
  return new Order();
 }
 public void updateOrder() {
  // TODO implement here
 }

 public void deleteOrder() {
  // TODO implement here
 }

 /**
     * Processes a refund for an order using payment gateway.
     */
    public void processRefund(String orderId, double amount) {
        // Call PaymentGateway to process refund (negative amount)
        paymentGateway.processPayment(orderId, -amount);
    }

    /**
     * Processes a discount for an order using payment gateway.
     */
    public void processDiscount(String orderId, double amount, double discountPercentage) {
        // Call PaymentGateway to process refund (negative amount)
        paymentGateway.processDiscount(orderId, amount, discountPercentage);
    }
    
    public Order placeOrder(Long userId, Long productId) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'placeOrder'");
    }
    public List<Order> getAllOrders() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getAllOrders'");
    }
}
