@Override
public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView view) {
    CsrfToken token = (CsrfToken) request.getAttribute(CsrfToken.class.getName());
    if (token != null) {
        view.addObject(token.getParameterName(), token);
    }
}