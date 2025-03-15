package com.poc.parser;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

public class GraphGenerator {
    public static void main(String[] args) throws Exception {

//        String pathToScan = "D:\\Gayatri\\BITS WILP\\Dissertation\\daytrader\\javaee6\\modules\\web\\src\\main\\java";
//        String packageName = "dayTrader";
//        String outputPath = "D:\\Gayatri\\BITS WILP\\Dissertation\\Parser\\dayTraderGraph2.json";
        String pathToScan = "D:\\Gayatri\\BITS WILP\\Dissertation\\POC2\\Monolith\\src\\main\\java";
        String packageName = "monolith.poc";
        String outputPath = "D:\\Gayatri\\BITS WILP\\Dissertation\\Parser\\ecommercePoc.json";
        
        // Set up symbol solver
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        // Replace with the actual path to Daytrader source code
        typeSolver.add(new JavaParserTypeSolver(pathToScan));
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);

        // Parse all Java files in the project
        Set<String> classNames = new HashSet<>();
        Set<String> resourceNames = new HashSet<>();
        JSONArray edges = new JSONArray();

        Files.walk(Paths.get(pathToScan)).forEach(path -> {
            if (path.toString().endsWith(".java")) {
                try {
                    CompilationUnit cu = StaticJavaParser.parse(new FileInputStream(path.toFile()));
                    // Extract class names
                    cu.findAll(ClassOrInterfaceDeclaration.class).forEach(clazz -> {
                        classNames.add(clazz.getNameAsString());
                    });
                    // Extract method calls
                    cu.findAll(MethodCallExpr.class).forEach(call -> {
                        try {
                            String callerClass = call.findAncestor(ClassOrInterfaceDeclaration.class)
                                    .map(ClassOrInterfaceDeclaration::getNameAsString)
                                    .orElse(null);
                            System.out.println("CallerClass is " +callerClass);
                            if (callerClass != null) {
                                String calleeClass = call.resolve().getClassName();
                                System.out.println("CalleeClass Name is "+calleeClass);
                                if(call.resolve().getPackageName().contains(packageName)) {
                                    JSONObject edge = new JSONObject()
                                            .put("from", callerClass)
                                            .put("to", calleeClass)
                                            .put("type", "calls");
                                    if (!edgeExists(edges, edge)) {
                                        edges.put(edge);
                                    }
                                }
                            }
                        } catch (Exception e) {
                            // Handle unresolved symbols
                        }
                    });
                    // Extract resource accesses (simplified for database accesses)
                    cu.findAll(MethodDeclaration.class).forEach(method -> {
                        if (method.toString().contains("java.sql")) {
                            String className = method.findAncestor(ClassOrInterfaceDeclaration.class)
                                    .map(ClassOrInterfaceDeclaration::getNameAsString)
                                    .orElse(null);
                            if (className != null) {
                                // Assume resource name extraction logic here
                                String resourceName = "DatabaseTable"; // Placeholder for actual table names
                                resourceNames.add(resourceName);
                                JSONObject edge = new JSONObject()
                                        .put("from", className)
                                        .put("to", resourceName)
                                        .put("type", "accesses");
                                if (!edgeExists(edges, edge)) {
                                    edges.put(edge);
                                }
                            }
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        // Create nodes
        JSONArray nodes = new JSONArray();
        for (String className : classNames) {
            nodes.put(new JSONObject().put("id", className).put("type", "class"));
        }
        for (String resourceName : resourceNames) {
            nodes.put(new JSONObject().put("id", resourceName).put("type", "resource"));
        }

        // Save to JSON
        JSONObject graph = new JSONObject();
        graph.put("nodes", nodes);
        graph.put("edges", edges);
        Files.write(Paths.get(outputPath), graph.toString().getBytes());
    }

    private static boolean edgeExists(JSONArray edges, JSONObject edge) {
        for (int i = 0; i < edges.length(); i++) {
            if (edges.getJSONObject(i).similar(edge)) {
                return true;
            }
        }
        return false;
    }
}
