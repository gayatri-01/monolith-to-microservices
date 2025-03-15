public void setSiteSlogan(String siteSlogan) {
    this.siteSlogan = siteSlogan;
    settingService.put(SITE_SLOGAN, siteSlogan);
}