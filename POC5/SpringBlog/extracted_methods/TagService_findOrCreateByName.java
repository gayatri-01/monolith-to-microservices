public Tag findOrCreateByName(String name) {
    Tag tag = tagRepository.findByName(name);
    if (tag == null) {
        tag = tagRepository.save(new Tag(name));
    }
    return tag;
}