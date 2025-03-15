public String getTagNames(Set<Tag> tags) {
    if (tags == null || tags.isEmpty()) {
        return "";
    }
    StringBuilder names = new StringBuilder();
    tags.forEach(tag -> names.append(tag.getName()).append(","));
    names.deleteCharAt(names.length() - 1);
    return names.toString();
}