@Override
public void addCorsMappings(CorsRegistry registry) {
    if (env.acceptsProfiles(ENV_DEVELOPMENT)) {
        log.debug("Register CORS configuration");
        registry.addMapping("/api/**").allowedOrigins("http://localhost:8080").allowedMethods("*").allowedHeaders("*").allowCredentials(true).maxAge(3600);
    }
}