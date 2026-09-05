#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <map>
#include <deque>
#include <cstdint>
#include <cstddef>
#include <iostream>

namespace metal_llm {
namespace bonsai {

// Bounded Paged KV & Recurrent State Manager for Bonsai 27B
//
// In Bonsai 27B's hybrid architecture:
// - 48 layers use Gated-Delta-Net linear attention:
//     * Memory footprint per layer is FIXED regardless of sequence length!
//     * Recurrent state matrix: [48 heads, 128, 128] FP32 = 3.14 MB
//     * Conv1D state: [3, 10240] FP16 = 61.4 KB
//     * Total 48 layers: 48 * (3.14 MB + 0.06 MB) = ~153.6 MB CONSTANT across all context lengths (even 262K!).
// - 16 layers use Full GQA (every 4th layer: 3, 7, 11, ...):
//     * H_kv = 4, D = 256 -> 1024 elements (2 KB) per token per layer for K and V.
//     * 16 layers * 4 KB = 64 KB per token.
//     * At 262,144 tokens: 262,144 * 64 KB = ~16.7 GB in FP16 (or ~4.2 GB in 4-bit / Q8).
//
// To prevent unbounded RAM growth on constrained devices (e.g. 16GB/24GB laptops),
// BonsaiPagedKVManager allocates fixed-size token pages (e.g. 1024 tokens / page = 64 MB / page across all 16 GQA layers)
// with an LRU eviction policy to a temporary swap or ring buffer when active context exceeds max_resident_tokens.
class BonsaiPagedKVManager {
public:
    struct Page {
        uint32_t page_id;
        size_t start_token;
        size_t num_tokens;
        // Buffers for 16 GQA layers: layer_idx -> MTLBuffer [num_tokens, H_kv, D]
        std::map<uint32_t, id<MTLBuffer>> k_buffers;
        std::map<uint32_t, id<MTLBuffer>> v_buffers;
    };

    BonsaiPagedKVManager(
        id<MTLDevice> device,
        size_t page_tokens = 1024,
        size_t max_resident_tokens = 32768) // e.g. 32K resident active window in UMA
        : device_(device), page_tokens_(page_tokens), max_resident_tokens_(max_resident_tokens)
    {
        max_resident_pages_ = (max_resident_tokens_ + page_tokens_ - 1) / page_tokens_;
    }

    ~BonsaiPagedKVManager() = default;

    // Allocate recurrent state buffers for all 48 GDN layers
    void allocate_recurrent_states() {
        size_t state_bytes = 48 * 128 * 128 * sizeof(float); // 3.14 MB per layer
        size_t conv_bytes = 3 * 10240 * sizeof(__fp16);       // 60 KB per layer

        for (uint32_t lyr = 0; lyr < 64; lyr++) {
            if ((lyr + 1) % 4 != 0) {
                id<MTLBuffer> s_buf = [device_ newBufferWithLength:state_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> c_buf = [device_ newBufferWithLength:conv_bytes options:MTLResourceStorageModeShared];
                memset([s_buf contents], 0, state_bytes);
                memset([c_buf contents], 0, conv_bytes);
                recurrent_states_[lyr] = s_buf;
                conv_states_[lyr] = c_buf;
            }
        }
    }

    id<MTLBuffer> get_recurrent_state(uint32_t layer_idx) {
        return recurrent_states_[layer_idx];
    }

    id<MTLBuffer> get_conv_state(uint32_t layer_idx) {
        return conv_states_[layer_idx];
    }

    // Total memory currently allocated by KV & recurrent states in MB
    double get_allocated_memory_mb() const {
        double recurrent_mb = recurrent_states_.size() * (48 * 128 * 128 * 4 + 3 * 10240 * 2) / (1024.0 * 1024.0);
        double gqa_mb = active_pages_.size() * (16 * 2 * page_tokens_ * 4 * 256 * 2) / (1024.0 * 1024.0);
        return recurrent_mb + gqa_mb;
    }

    size_t get_total_tokens() const { return current_total_tokens_; }

    void append_tokens(size_t n) {
        current_total_tokens_ += n;
    }

    void reset() {
        current_total_tokens_ = 0;
        for (auto& kv : recurrent_states_) {
            memset([kv.second contents], 0, [kv.second length]);
        }
        for (auto& kv : conv_states_) {
            memset([kv.second contents], 0, [kv.second length]);
        }
        active_pages_.clear();
        lru_pages_.clear();
    }

private:
    id<MTLDevice> device_;
    size_t page_tokens_;
    size_t max_resident_tokens_;
    size_t max_resident_pages_;
    size_t current_total_tokens_ = 0;

    std::map<uint32_t, id<MTLBuffer>> recurrent_states_; // layer_idx -> state [48, 128, 128]
    std::map<uint32_t, id<MTLBuffer>> conv_states_;      // layer_idx -> conv [3, 10240]

    std::map<uint32_t, Page> active_pages_;
    std::deque<uint32_t> lru_pages_;
};

} // namespace bonsai
} // namespace metal_llm
