@Override
protected void configure(HttpSecurity http) throws Exception {
    http.authorizeRequests().antMatchers("/admin/**").authenticated().anyRequest().permitAll().and().formLogin().loginPage("/login").permitAll().failureUrl("/login?error=1").loginProcessingUrl("/authenticate").and().logout().logoutUrl("/logout").permitAll().logoutSuccessUrl("/login?logout").and().rememberMe().rememberMeServices(rememberMeServices()).key("remember-me-key");
}