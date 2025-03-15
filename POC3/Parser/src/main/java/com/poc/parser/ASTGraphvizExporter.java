package com.poc.parser;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;

public class ASTGraphvizExporter {
    public static void main(String[] args) throws Exception {
        String filePath = "D:\\Gayatri\\BITS WILP\\Dissertation\\POC2\\Monolith\\src\\main\\java\\com\\monolith\\poc\\PocApplication.java";
        CompilationUnit cu = StaticJavaParser.parse(new FileInputStream(filePath));

        StringBuilder dot = new StringBuilder();
        dot.append("digraph AST {\n");
        AtomicInteger nodeCounter = new AtomicInteger(0);

        generateDot(cu, dot, nodeCounter, -1);
        dot.append("}\n");

        // Save to file
        Files.write(Paths.get("ast_graph.dot"), dot.toString().getBytes());
        System.out.println("DOT file generated: ast_graph.dot");
    }

    private static int generateDot(Node node, StringBuilder dot, AtomicInteger nodeCounter, int parent) {
        int currentNode = nodeCounter.incrementAndGet();
        dot.append("  node").append(currentNode).append(" [label=\"").append(node.getClass().getSimpleName()).append("\"];\n");

        if (parent != -1) {
            dot.append("  node").append(parent).append(" -> node").append(currentNode).append(";\n");
        }

        for (Node child : node.getChildNodes()) {
            generateDot(child, dot, nodeCounter, currentNode);
        }
        return currentNode;
    }
}
