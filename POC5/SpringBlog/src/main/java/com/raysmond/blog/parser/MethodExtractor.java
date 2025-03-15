package com.raysmond.blog.parser;


import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class MethodExtractor {

    static List<String> methodFiles = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        // Specify the root package (directory) to start traversal
//        String packagePath = "src\\main\\java\\com\\raysmond\\blog\\repositories";// Change as needed
//        String packagePath = "src\\main\\java\\com\\raysmond\\blog\\services";// Change as needed
        String packagePath = "src\\main\\java\\com\\raysmond\\blog\\security";// Change as needed

        // Output directory for extracted methods
        String outputDir = "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\";
        Files.createDirectories(Paths.get(outputDir));

        // Start recursive traversal
        processDirectory(new File(packagePath), outputDir);

        System.out.println(methodFiles);
    }

    private static void processDirectory(File directory, String outputDir) throws IOException {
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Invalid package path: " + directory.getAbsolutePath());
            return;
        }

        JavaParser parser = new JavaParser();

        for (File file : Objects.requireNonNull(directory.listFiles())) {
            if (file.isDirectory()) {
                // Recursively process subdirectories
                processDirectory(file, outputDir);
            } else if (file.getName().endsWith(".java")) {
                System.out.println("Processing: " + file.getAbsolutePath());

                // Parse Java file
                CompilationUnit cu = parser.parse(file).getResult().orElseThrow(
                        () -> new IOException("Error parsing " + file.getName())
                );

                // Extract and save methods
                cu.accept(new MethodVisitor(outputDir, file.getName()), null);
            }
        }
    }

    // Visitor to extract methods
    private static class MethodVisitor extends VoidVisitorAdapter<Void> {
        private final String outputDir;
        private final String className;

        public MethodVisitor(String outputDir, String className) {
            this.outputDir = outputDir;
            this.className = className.replace(".java", ""); // Remove .java extension
        }

        @Override
        public void visit(MethodDeclaration method, Void arg) {
            super.visit(method, arg);

            // Get method name
            String methodName = method.getNameAsString();

            // Get method code
            String methodCode = method.toString();

            // Ensure method names are unique by adding class name
            String filePath = outputDir + className + "_" + methodName + ".java";

            // Save method as a separate Java file
            try {
                Files.write(Paths.get(filePath), methodCode.getBytes());
                System.out.println("Extracted: " + filePath);
                methodFiles.add("\""+filePath+"\"");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}

