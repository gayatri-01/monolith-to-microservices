package com.monolith.poc.service;


public class PaymentGateway {
    /**
     * Processes a payment for the given user and amount.
     * @param userId The ID of the user making the payment.
     * @param amount The amount to be paid.
     * @return true if payment is successful, false otherwise.
     */
    public boolean processPayment(String userId, double amount) {
        // Simulate payment processing logic
        System.out.println("Processing payment of " + amount + " for user " + userId);
        return true;  // Assume payment always succeeds for simplicity
    }

    public void processDiscount(String userId, double amount, double discountPercentage) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'processDiscount'");
    }

    
}
