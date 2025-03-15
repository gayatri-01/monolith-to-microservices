public String getSiteName() {
    return (String) settingService.get(SITE_NAME, siteName);
}