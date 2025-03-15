@Bean
public HandlerInterceptor viewObjectAddingInterceptor() {
    return new HandlerInterceptorAdapter() {

        @Override
        public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
            viewHelper.setStartTime(System.currentTimeMillis());
            return true;
        }

        @Override
        public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView view) {
            CsrfToken token = (CsrfToken) request.getAttribute(CsrfToken.class.getName());
            if (token != null) {
                view.addObject(token.getParameterName(), token);
            }
        }
    };
}