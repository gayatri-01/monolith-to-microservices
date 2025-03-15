package com.monolith.poc.model;


import jakarta.persistence.*;
import lombok.*;
import com.monolith.poc.model.User;
import com.monolith.poc.model.Product;

@Entity
@Getter @Setter @NoArgsConstructor @AllArgsConstructor
@Table(name = "\"order\"")
public class Order {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    private User user;

    @ManyToOne
    private Product product;
}

