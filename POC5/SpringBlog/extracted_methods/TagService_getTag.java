public Tag getTag(String tagName) {
    return tagRepository.findByName(tagName);
}