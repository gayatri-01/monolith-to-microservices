public String getSiteSlogan() {
    return (String) settingService.get(SITE_SLOGAN, siteSlogan);
}