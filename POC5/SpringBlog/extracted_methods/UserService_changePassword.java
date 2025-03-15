public boolean changePassword(User user, String password, String newPassword) {
    if (password == null || newPassword == null || password.isEmpty() || newPassword.isEmpty())
        return false;
    boolean match = passwordEncoder.matches(password, user.getPassword());
    if (!match)
        return false;
    user.setPassword(passwordEncoder.encode(newPassword));
    userRepository.save(user);
    log.info("User @{} changed password.", user.getEmail());
    return true;
}