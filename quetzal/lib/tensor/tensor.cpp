#include "tensor/tensor.hpp"

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace quetzal::tensor {

static void* aligned_alloc_wrapper(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void aligned_free_wrapper(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static std::size_t alloc_count = 0;

void clear_alloc_count() {
    alloc_count = 0;
}

std::size_t get_alloc_count() {
    return alloc_count;
}

void* default_allocator(size_t size) {
    alloc_count++;
    return aligned_alloc_wrapper(size, alignof(std::max_align_t));
}

void default_deallocator(void* ptr) {
    if (ptr != nullptr) {
        aligned_free_wrapper(ptr);
    }
}

}
