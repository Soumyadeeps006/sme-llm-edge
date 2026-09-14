#pragma once
#include <string>
#include <cstdint>

struct MappedFile {
    void* data;
    size_t size;
    int fd;
};

// Maps a file into memory. Returns true on success.
bool mmap_load(const std::string& filepath, MappedFile& out_mapping);
void mmap_unload(MappedFile& mapping);