@PostConstruct
public void registerJadeViewHelpers() {
    viewHelper.setApplicationEnv(this.getApplicationEnv());
}