public Post createAboutPage() {
    log.debug("Create default about page");
    Post post = new Post();
    post.setTitle(Constants.ABOUT_PAGE_PERMALINK);
    post.setContent(Constants.ABOUT_PAGE_PERMALINK.toLowerCase());
    post.setPermalink(Constants.ABOUT_PAGE_PERMALINK);
    post.setUser(userService.getSuperUser());
    post.setPostFormat(PostFormat.MARKDOWN);
    return createPost(post);
}