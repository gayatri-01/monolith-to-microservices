public Page<Post> getAllPublishedPostsByPage(int page, int pageSize) {
    log.debug("Get posts by page " + page);
    return postRepository.findAllByPostTypeAndPostStatus(PostType.POST, PostStatus.PUBLISHED, new PageRequest(page, pageSize, Sort.Direction.DESC, "createdAt"));
}