public Post getPublishedPostByPermalink(String permalink) {
    log.debug("Get post with permalink " + permalink);
    Post post = postRepository.findByPermalinkAndPostStatus(permalink, PostStatus.PUBLISHED);
    if (post == null) {
        try {
            post = postRepository.findOne(Long.valueOf(permalink));
        } catch (NumberFormatException e) {
            post = null;
        }
    }
    if (post == null) {
        throw new NotFoundException("Post with permalink '" + permalink + "' is not found.");
    }
    return post;
}