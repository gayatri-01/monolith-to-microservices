public User currentUser() {
    Authentication auth = SecurityContextHolder.getContext().getAuthentication();
    if (auth == null || auth instanceof AnonymousAuthenticationToken) {
        return null;
    }
    String email = ((org.springframework.security.core.userdetails.User) auth.getPrincipal()).getUsername();
    return userRepository.findByEmail(email);
}