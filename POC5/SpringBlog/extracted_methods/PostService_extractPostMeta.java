private Post extractPostMeta(Post post) {
    Post archivePost = new Post();
    archivePost.setId(post.getId());
    archivePost.setTitle(post.getTitle());
    archivePost.setPermalink(post.getPermalink());
    archivePost.setCreatedAt(post.getCreatedAt());
    return archivePost;
}