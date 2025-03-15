public Post createPost(Post post) {
    if (post.getPostFormat() == PostFormat.MARKDOWN) {
        post.setRenderedContent(markdownService.renderToHtml(post.getContent()));
        post.setRenderedSummary(markdownService.renderToHtml(post.getSummary()));
    }
    return postRepository.save(post);
}