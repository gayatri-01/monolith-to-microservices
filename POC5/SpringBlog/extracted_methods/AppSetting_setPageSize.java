public void setPageSize(Integer pageSize) {
    this.pageSize = pageSize;
    settingService.put(PAGE_SIZE, pageSize);
}