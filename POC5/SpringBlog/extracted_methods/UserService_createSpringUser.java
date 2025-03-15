private org.springframework.security.core.userdetails.User createSpringUser(User user) {
    return new org.springframework.security.core.userdetails.User(user.getEmail(), user.getPassword(), Collections.singleton(createAuthority(user)));
}