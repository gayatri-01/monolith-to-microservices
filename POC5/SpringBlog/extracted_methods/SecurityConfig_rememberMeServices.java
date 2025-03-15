@Bean
public TokenBasedRememberMeServices rememberMeServices() {
    return new TokenBasedRememberMeServices("remember-me-key", userService);
}