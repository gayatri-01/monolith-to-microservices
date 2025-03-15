@Override
@Cacheable(value = "settingCache", key = "#key")
public Serializable get(String key, Serializable defaultValue) {
    Serializable value = get(key);
    return value == null ? defaultValue : value;
}