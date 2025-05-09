#if 0

#pragma once

// 暂时不实现，以后需要的时候再实现.
#include <cstdint>
#include <memory>
#include <vector>

struct llama_file;
struct llama_mmap;
// 加载模型如果不是多线程的话暂时不用lock
// struct llama_mlock;

using llama_files = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps = std::vector<std::unique_ptr<llama_mmap>>;
// using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char* fname, const char* mode);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void* ptr, size_t len) const;
    uint32_t read_u32() const;

    void write_raw(const void* ptr, size_t len) const;
    void write_u32(uint32_t val) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap&) = delete;
    llama_mmap(struct llama_file* file, size_t prefetch = (size_t)-1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void* addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
#endif

