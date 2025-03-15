package com.monolith.poc.controller;

import com.monolith.poc.model.Cart;
import com.monolith.poc.model.CartItem;
import com.monolith.poc.model.Order;
import com.monolith.poc.service.CartService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cart")
public class CartController {

    private  CartService cartService;

    @Autowired
    public CartController(CartService cartService) {
        this.cartService = cartService;
    }

    @PostMapping("/add")
    public void addItemToCart(@RequestParam Long userId, @RequestBody CartItem cartItem) {

        cartService.addItemToCart(userId, cartItem);
    }

    @GetMapping("/{userId}")
    public Cart getCart(@PathVariable Long userId) {
        return cartService.getCartByUserId(userId);
    }

    @DeleteMapping("/remove")
    public Cart removeItemFromCart(@RequestParam Long userId, @RequestParam Long itemId) {
        return cartService.removeItemFromCart(userId, itemId);
    }

    @PostMapping("/checkout")
    public Order checkout(@RequestParam Long userId, @RequestBody List<CartItem> items) {
        return cartService.checkout(userId, items);
    }
}