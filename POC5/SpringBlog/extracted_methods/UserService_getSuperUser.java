public User getSuperUser() {
    User user = userRepository.findByEmail(Constants.DEFAULT_ADMIN_EMAIL);
    if (user == null) {
        user = createUser(new User(Constants.DEFAULT_ADMIN_EMAIL, Constants.DEFAULT_ADMIN_PASSWORD, User.ROLE_ADMIN));
    }
    return user;
}