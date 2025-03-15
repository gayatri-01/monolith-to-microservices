public void signin(User user) {
    SecurityContextHolder.getContext().setAuthentication(authenticate(user));
}