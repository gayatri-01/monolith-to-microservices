package com.monolith;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.comments.Comment;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class CommentExtractor {
    public static List<String> extractComments(String projectPath) throws Exception {
        File dir = new File(projectPath);
        List<String> allComments = new ArrayList<>();

        for (File file : dir.listFiles((d, name) -> name.endsWith(".java"))) {
            CompilationUnit cu = StaticJavaParser.parse(file);
            List<Comment> comments = cu.getAllContainedComments();
            allComments.addAll(comments.stream().map(Comment::getContent).collect(Collectors.toList()));
        }

        return allComments;
    }

    public static void main(String[] args) throws Exception {
        List<String> comments = extractComments("D:\\Gayatri\\BITS WILP\\Dissertation\\LLMs\\Monolith\\src\\main\\java\\com\\monolith\\poc\\service");
        comments.forEach(System.out::println);
    }
}