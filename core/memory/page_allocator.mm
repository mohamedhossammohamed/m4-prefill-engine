#include "core/memory/page_allocator.h"
#include <sys/mman.h>
#include <new>

namespace core::memory {

void* allocate_16kb_aligned(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    // Round user bytes up to a multiple of 16KB system page size
    size_t aligned_user_bytes = (bytes + SYSTEM_PAGE_SIZE_16KB - 1) & ~(SYSTEM_PAGE_SIZE_16KB - 1);
    // Prepend a 16KB metadata page to record the total mapping size for munmap
    size_t total_alloc_bytes = aligned_user_bytes + SYSTEM_PAGE_SIZE_16KB;

    void* raw = mmap(nullptr, total_alloc_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (raw == MAP_FAILED || raw == nullptr) {
        throw std::bad_alloc();
    }

    *reinterpret_cast<size_t*>(raw) = total_alloc_bytes;
    void* user_ptr = static_cast<char*>(raw) + SYSTEM_PAGE_SIZE_16KB;
    assert_16kb_aligned(user_ptr);
    return user_ptr;
}

void free_16kb_aligned(void* ptr) {
    if (ptr) {
        void* raw = static_cast<char*>(ptr) - SYSTEM_PAGE_SIZE_16KB;
        size_t total_alloc_bytes = *reinterpret_cast<size_t*>(raw);
        munmap(raw, total_alloc_bytes);
    }
}

} // namespace core::memory
