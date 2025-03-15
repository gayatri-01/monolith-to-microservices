package com.monolith.poc.model;


import jakarta.persistence.*;
import lombok.*;

import java.util.List;

@Entity
@Getter @Setter @NoArgsConstructor @AllArgsConstructor
@Table(name = "\"order\"")
public class Order {
    public Order(String userId, List<CartItem> items) {
        //TODO Auto-generated constructor stub
    }

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    private User user;

    @ManyToOne
    private List<CartItem> cartItems;
}

