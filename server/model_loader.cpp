#include "model_loader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

bool mmap_load(const std::string& filepath, MappedFile& out_mapping) {
    out_mapping.fd = open(filepath.c_str(), O_RDONLY);
    if (out_mapping.fd == -1) {
        std::cerr << "Failed to open file for mmap: " << filepath << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(out_mapping.fd, &sb) == -1) {
        close(out_mapping.fd);
        return false;
    }
    out_mapping.size = sb.st_size;

    // MAP_PRIVATE ensures changes aren't written to disk. 
    // MAP_POPULATE pre-faults pages for faster initial access on Linux.
    out_mapping.data = mmap(nullptr, out_mapping.size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, out_mapping.fd, 0);
    
    if (out_mapping.data == MAP_FAILED) {
        close(out_mapping.fd);
        std::cerr << "mmap failed!" << std::endl;
        return false;
    }
    
    std::cout << "Successfully mmap'd " << (out_mapping.size / 1024 / 1024) << " MB." << std::endl;
    return true;
}

void mmap_unload(MappedFile& mapping) {
    if (mapping.data && mapping.data != MAP_FAILED) {
        munmap(mapping.data, mapping.size);
    }
    if (mapping.fd != -1) {
        close(mapping.fd);
    }
}