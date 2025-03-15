@Override
public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
    viewHelper.setStartTime(System.currentTimeMillis());
    return true;
}