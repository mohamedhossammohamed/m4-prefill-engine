#pragma once

#include <cstddef>
#import <Metal/Metal.h>

namespace core::memory {

constexpr size_t DEFAULT_CACHE_PURGE_BYTES = 32 * 1024 * 1024; // 32MB (> 24MB M4 SLC)

// Pre-benchmark 32MB Direct I/O Unified Buffer Cache (UBC) dummy file purge
// Writes and reads a 32MB 16KB-aligned buffer with F_NOCACHE and fsync (Invariant 2).
void purge_unified_buffer_cache(const char* dummy_path = "/tmp/ubc_purge_dummy", size_t bytes = DEFAULT_CACHE_PURGE_BYTES);

// Pre-benchmark 32MB CPU XOR memory sweep to evict CPU L1/L2 and System-Level Cache (SLC).
// If buffer is null, an internal aligned buffer is used.
void cold_cache_evict_cpu(void* buffer = nullptr, size_t bytes = DEFAULT_CACHE_PURGE_BYTES);

// Pre-benchmark GPU SLC cache eviction via shared buffer fill/dispatch
void purge_slc_cache_gpu(id<MTLDevice> device, id<MTLCommandQueue> queue, size_t bytes = DEFAULT_CACHE_PURGE_BYTES);

// Combined cold-cache purge: UBC Direct I/O purge + CPU XOR sweep (+ optional GPU purge if device/queue supplied)
void purge_all_cold_caches(id<MTLDevice> device = nil, id<MTLCommandQueue> queue = nil, const char* dummy_path = "/tmp/ubc_purge_dummy");

} // namespace core::memory
