public List<Object[]> countPostsByTags() {
    log.debug("Count posts group by tags.");
    return postRepository.countPostsByTags(PostStatus.PUBLISHED);
}