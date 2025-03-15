package com.monolith.poc.service;

import com.monolith.poc.model.*;
import com.monolith.poc.repository.OrderRepository;
import com.monolith.poc.repository.PaymentRepository;
import com.monolith.poc.repository.ProductRepository;
import com.monolith.poc.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ECommerceService {
    private final UserRepository userRepository;
    private final ProductRepository productRepository;
    private final OrderRepository orderRepository;
    private final PaymentRepository paymentRepository;

    public ECommerceService(UserRepository userRepository, ProductRepository productRepository,
                            OrderRepository orderRepository, PaymentRepository paymentRepository) {
        this.userRepository = userRepository;
        this.productRepository = productRepository;
        this.orderRepository = orderRepository;
        this.paymentRepository = paymentRepository;
    }

    // User Operations
    public User createUser(User user) {
        return userRepository.save(user);
    }
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    // Product Operations
    public Product createProduct(Product product) {
        return productRepository.save(product);
    }
    public List<Product> getAllProducts() {
        return productRepository.findAll();
    }

    // Order Operations
    public Order placeOrder(Long userId, Long productId) {
        User user = userRepository.findById(userId).orElseThrow();
        Product product = productRepository.findById(productId).orElseThrow();
        Order order = new Order(null, user, product);
        return orderRepository.save(order);
    }

    public List<Order> getAllOrders() {
        return orderRepository.findAll();
    }

    // Payment Operations
    public Payment processPayment(Long orderId) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        Payment payment = new Payment(null, order, "Completed");
        return paymentRepository.save(payment);
    }
}
