@Bean
public PasswordEncoder passwordEncoder() {
    return new StandardPasswordEncoder();
}