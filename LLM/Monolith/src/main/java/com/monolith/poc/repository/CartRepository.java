package com.monolith.poc.repository;

import com.monolith.poc.model.Cart;
import com.monolith.poc.model.CartItem;
import com.monolith.poc.model.Order;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface CartRepository extends JpaRepository<Cart, Long> {

    List<CartItem> findByUserId(String userId);

    void clearCart(String userId);}
