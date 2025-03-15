private GrantedAuthority createAuthority(User user) {
    return new SimpleGrantedAuthority(user.getRole());
}