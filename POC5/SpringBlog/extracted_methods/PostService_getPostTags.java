public List<Tag> getPostTags(Post post) {
    log.debug("Get tags of post {}", post.getId());
    List<Tag> tags = new ArrayList<>();
    // Load the post first. If not, when the post is cached before while the tags not,
    // then the LAZY loading of post tags will cause an initialization error because
    // of not hibernate connection session
    postRepository.findOne(post.getId()).getTags().forEach(tags::add);
    return tags;
}