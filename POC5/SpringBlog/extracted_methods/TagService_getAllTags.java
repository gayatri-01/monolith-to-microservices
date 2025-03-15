public List<Tag> getAllTags() {
    return tagRepository.findAll();
}