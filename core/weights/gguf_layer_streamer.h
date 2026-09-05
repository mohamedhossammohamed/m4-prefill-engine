#pragma once

// GGUFLayerStreamer: opt-in layer-by-layer (block-group) weight serving.
//
// Wraps GGUFLoader (whole-file mmap, zero-copy) and adds a bounded-residency
// serving policy: only `resident_layer_count` transformer layers are logically
// resident at once. Acquiring a layer prefetches the next (MADV_WILLNEED) and
// evicts the least-recently-used resident layer (MADV_DONTNEED).
//
// ADDITIVE ONLY: does not change GGUFLoader, any quantization codec
// (Q4_0, MLX_4BIT, Q4_K, TERNARY_1_58, VAR_RATE_AFFINE, EXL3, PRISM_Q2_0),
// or any Metal kernel. Acquired pointers alias the loader mmap; the existing
// in-memory `unpack_column` paths consume them unchanged.
//
// NOTE (Bonsai 27B topology gap): this streamer serves WEIGHT BUFFERS only.
// It does not make the uniform-GQA `TransformerLayerCoordinator` capable of
// gated-delta-net linear attention (48/64 Bonsai layers). See report.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "core/weights/gguf_loader.h"

namespace metal_llm {

struct LayerGroup {
    int layer_index = -2; // -1 = shared (non-layer) tensors, >=0 = transformer layer
    std::vector<std::string> tensor_names;
    size_t bytes = 0;
};

class GGUFLayerStreamer {
public:
    GGUFLayerStreamer() = default;
    ~GGUFLayerStreamer() { close(); }

    GGUFLayerStreamer(const GGUFLayerStreamer&) = delete;
    GGUFLayerStreamer& operator=(const GGUFLayerStreamer&) = delete;

    // Opens a GGUF file for streaming. resident_layer_count >= 1 controls how
    // many transformer layers stay resident (shared tensors always resident).
    bool open(const std::string& filepath, size_t resident_layer_count = 2);
    void close();
    bool is_open() const { return is_open_; }

    size_t num_layers() const { return layer_order_.size(); }
    std::vector<int> layer_indices() const { return layer_order_; }
    size_t resident_layer_count() const { return resident_layer_count_; }

    // Tensor names for a layer (or shared group with layer_index == -1).
    std::vector<std::string> layer_tensor_names(int layer_index) const;

    // Makes a layer resident (prefetch next, evict LRU beyond budget).
    // Returns false for unknown layer_index.
    bool acquire_layer(int layer_index);

    // Zero-copy pointer into the loader mmap. Valid while streamer is open.
    const uint8_t* tensor_data(const std::string& name) const;

    // Residency accounting (logical bytes under management, not OS RSS).
    size_t current_resident_bytes() const { return current_resident_bytes_; }
    size_t peak_resident_bytes() const { return peak_resident_bytes_; }
    size_t total_model_bytes() const { return total_model_bytes_; }
    size_t shared_bytes() const { return shared_bytes_; }

    const GGUFLoader& loader() const { return loader_; }

    // Parses "model.layers.N." prefix; returns N or -1 for shared tensors.
    static int parse_layer_index(const std::string& tensor_name);

private:
    size_t tensor_byte_size(const GGUFTensorInfo& info) const;
    void advise_range(const uint8_t* ptr, size_t len, int advice) const;
    void evict_if_over_budget();

    bool is_open_ = false;
    size_t resident_layer_count_ = 2;

    GGUFLoader loader_;
    std::map<int, LayerGroup> groups_;          // layer_index -> group
    std::vector<int> layer_order_;              // sorted layer indices (>= 0)
    std::map<std::string, size_t> tensor_bytes_;

    std::deque<int> resident_lru_;              // front = LRU
    std::set<int> resident_set_;

    size_t current_resident_bytes_ = 0;
    size_t peak_resident_bytes_ = 0;
    size_t total_model_bytes_ = 0;
    size_t shared_bytes_ = 0;
};

} // namespace metal_llm
