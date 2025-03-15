@Async
public void incrementViews(Long postId) {
    synchronized (this) {
        Post post = postRepository.findOne(postId);
        post.setViews(post.getViews() + 1);
        postRepository.save(post);
    }
}