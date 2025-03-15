@Override
public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
    User user = userRepository.findByEmail(username);
    if (user == null) {
        throw new UsernameNotFoundException("user not found");
    }
    return createSpringUser(user);
}