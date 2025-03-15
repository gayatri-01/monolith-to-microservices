@Override
@CacheEvict(value = "settingCache", key = "#key")
public void put(String key, Serializable value) {
    log.info("Update setting " + key + " to database. Value = " + value);
    Setting setting = settingRepository.findByKey(key);
    if (setting == null) {
        setting = new Setting();
        setting.setKey(key);
    }
    try {
        setting.setValue(value);
        settingRepository.save(setting);
    } catch (Exception ex) {
        log.info("Cannot save setting value with type: " + value.getClass() + ". key = " + key);
    }
}