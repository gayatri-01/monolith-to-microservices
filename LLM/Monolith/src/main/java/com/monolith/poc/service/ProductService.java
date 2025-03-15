package com.monolith.poc.service;

import com.monolith.poc.model.Product;
import org.springframework.stereotype.Service;

import com.monolith.poc.exeption.UnauthorizedException;
import java.util.Collections;
import java.util.List;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;

@Service
public class ProductService {

    public Product createProduct(Product product) {
        // Implementation here
        return new Product();
    }

    public List<Product> getAllProducts() {
        // Implementation here
        return Collections.emptyList();
    }

    public Product getProductById(Long productId) {
        // Implementation here
        return new Product();
    }


    public void deleteProduct(Long productId) {
        // Implementation here
    }

    // Verify JWT token and check for admin role
    private boolean verifyAdminToken(String token) {
        try {
            Claims claims = Jwts.parserBuilder()
            .build()
            .parseClaimsJws(token)
            .getBody();
            String role = claims.get("role", String.class);
            return "admin".equals(role);
        } catch (Exception e) {
            return false; // Invalid signature
        }
    }

    // Update product, requiring admin authentication
    public void updateProduct(Long productId, Product product) {
        if (!verifyAdminToken(getTokenFromHeader())) {
            throw new UnauthorizedException("Invalid or missing admin token");
        }
        // Logic to update product
        //....
    }


     // Dummy method to get token from header
     private String getTokenFromHeader() {
        // Dummy implementation to get token from header
        return "dummyToken";
    }
}