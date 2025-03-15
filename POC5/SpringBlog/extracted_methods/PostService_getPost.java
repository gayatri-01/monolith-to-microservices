public Post getPost(Long postId) {
    log.debug("Get post " + postId);
    Post post = postRepository.findOne(postId);
    if (post == null) {
        throw new NotFoundException("Post with id " + postId + " is not found.");
    }
    return post;
}