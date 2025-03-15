@Override
public void addInterceptors(InterceptorRegistry registry) {
    registry.addInterceptor(viewObjectAddingInterceptor());
    super.addInterceptors(registry);
}