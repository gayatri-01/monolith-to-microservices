// cache or not?
public Page<Post> findPostsByTag(String tagName, int page, int pageSize) {
    return postRepository.findByTag(tagName, new PageRequest(page, pageSize, Sort.Direction.DESC, "createdAt"));
}