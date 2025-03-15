package com.monolith.poc.model;

import jakarta.persistence.*;
import lombok.*;
import com.monolith.poc.model.Order;

@Entity
@Getter @Setter @NoArgsConstructor @AllArgsConstructor
public class Payment {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne
    private Order order;

    private String status;
}

