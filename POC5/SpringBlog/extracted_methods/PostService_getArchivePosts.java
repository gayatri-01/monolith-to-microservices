public List<Post> getArchivePosts() {
    log.debug("Get all archive posts from database.");
    Pageable page = new PageRequest(0, Integer.MAX_VALUE, Sort.Direction.DESC, "createdAt");
    return postRepository.findAllByPostTypeAndPostStatus(PostType.POST, PostStatus.PUBLISHED, page).getContent().stream().map(this::extractPostMeta).collect(Collectors.toList());
}