#include "core/memory/cache_flush.h"
#include "core/memory/page_allocator.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <memory>

namespace core::memory {

void purge_unified_buffer_cache(const char* dummy_path, size_t bytes) {
    if (!dummy_path || bytes == 0) {
        return;
    }

    // Ensure bytes is a multiple of 16KB for Direct I/O
    size_t aligned_bytes = (bytes + SYSTEM_PAGE_SIZE_16KB - 1) & ~(SYSTEM_PAGE_SIZE_16KB - 1);
    void* dummy = allocate_16kb_aligned(aligned_bytes);
    std::memset(dummy, 0x5A, aligned_bytes);

    int fd = open(dummy_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
        fcntl(fd, F_NOCACHE, 1);
        pwrite(fd, dummy, aligned_bytes, 0);
        fsync(fd);
        pread(fd, dummy, aligned_bytes, 0);
        close(fd);
        unlink(dummy_path);
    }
    free_16kb_aligned(dummy);
}

void cold_cache_evict_cpu(void* buffer, size_t bytes) {
    if (bytes == 0) {
        return;
    }

    std::unique_ptr<uint8_t, void(*)(void*)> local_buf(nullptr, free_16kb_aligned);
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    if (!ptr) {
        ptr = static_cast<uint8_t*>(allocate_16kb_aligned(bytes));
        std::memset(ptr, 0x5A, bytes);
        local_buf.reset(ptr);
    }

    volatile uint32_t cpu_sum = 0;
    uint32_t* u32_ptr = reinterpret_cast<uint32_t*>(ptr);
    size_t u32_count = bytes / sizeof(uint32_t);

    // Stride by 16 elements (64 bytes = 1 cacheline)
    for (size_t i = 0; i < u32_count; i += 16) {
        u32_ptr[i] ^= static_cast<uint32_t>(i * 1664525u + 1013904223u);
        cpu_sum += u32_ptr[i];
    }
    (void)cpu_sum;
}

void purge_slc_cache_gpu(id<MTLDevice> device, id<MTLCommandQueue> queue, size_t bytes) {
    if (!device || !queue || bytes == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> gpu_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (gpu_buf) {
            uint32_t* contents = static_cast<uint32_t*>([gpu_buf contents]);
            size_t u32_count = bytes / sizeof(uint32_t);
            for (size_t i = 0; i < u32_count; i += 16) {
                contents[i] ^= static_cast<uint32_t>(i * 1664525u + 1013904223u);
            }
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit fillBuffer:gpu_buf range:NSMakeRange(0, bytes) value:0x5A];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            gpu_buf = nil;
        }
    }
}

void purge_all_cold_caches(id<MTLDevice> device, id<MTLCommandQueue> queue, const char* dummy_path) {
    purge_unified_buffer_cache(dummy_path, DEFAULT_CACHE_PURGE_BYTES);
    cold_cache_evict_cpu(nullptr, DEFAULT_CACHE_PURGE_BYTES);
    if (device && queue) {
        purge_slc_cache_gpu(device, queue, DEFAULT_CACHE_PURGE_BYTES);
    }
}

} // namespace core::memory
