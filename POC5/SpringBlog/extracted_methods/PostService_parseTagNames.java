public Set<Tag> parseTagNames(String tagNames) {
    Set<Tag> tags = new HashSet<>();
    if (tagNames != null && !tagNames.isEmpty()) {
        tagNames = tagNames.toLowerCase();
        String[] names = tagNames.split("\\s*,\\s*");
        for (String name : names) {
            tags.add(tagService.findOrCreateByName(name));
        }
    }
    return tags;
}