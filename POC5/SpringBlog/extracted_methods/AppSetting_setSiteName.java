public void setSiteName(String siteName) {
    this.siteName = siteName;
    settingService.put(SITE_NAME, siteName);
}