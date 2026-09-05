#include "gguf_layer_streamer.h"
#include <algorithm>
#include <iostream>
#include <sys/mman.h>

// MADV_WILLNEED / MADV_DONTNEED exist on macOS; guard for portability.
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif

namespace metal_llm {

int GGUFLayerStreamer::parse_layer_index(const std::string& name) {
    const std::string prefix = "model.layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) return -1;
    size_t pos = prefix.size();
    size_t end = pos;
    while (end < name.size() && name[end] >= '0' && name[end] <= '9') end++;
    if (end == pos || end >= name.size() || name[end] != '.') return -1;
    return std::stoi(name.substr(pos, end - pos));
}

size_t GGUFLayerStreamer::tensor_byte_size(const GGUFTensorInfo& info) const {
    if (info.dims.empty()) return 0;
    uint64_t total_elements = 1;
    for (uint64_t dim : info.dims) total_elements *= dim;
    if (total_elements == 0) return 0;
    // PrismML Q2_0 (and test fixtures using type id 100): 128 elems / 34 bytes.
    // Generic Q2_0-family fallback for any tensor with 128-divisible shape.
    if ((total_elements % 128) == 0) {
        return (size_t)(total_elements / 128) * 34;
    }
    return 0; // Unknown layout: untracked (pointer serving still works).
}

void GGUFLayerStreamer::advise_range(const uint8_t* ptr, size_t len, int advice) const {
    if (!ptr || len == 0) return;
    const size_t page = 16384; // Apple Silicon system page for mmap advice
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = addr & ~(page - 1);
    size_t adj_len = (size_t)(addr - aligned) + len;
    adj_len = ((adj_len + page - 1) / page) * page;
    // Best-effort: ignore failures (e.g. buffers not yet faulted in).
    madvise(reinterpret_cast<void*>(aligned), adj_len, advice);
}

bool GGUFLayerStreamer::open(const std::string& filepath, size_t resident_layer_count) {
    close();
    if (resident_layer_count < 1) resident_layer_count = 1;
    resident_layer_count_ = resident_layer_count;

    if (!loader_.open(filepath)) return false;

    // Group tensors by layer index.
    for (const std::string& name : loader_.tensor_names()) {
        const GGUFTensorInfo* info = loader_.get_tensor_info(name);
        if (!info) continue;
        int idx = parse_layer_index(name);
        size_t bytes = tensor_byte_size(*info);
        tensor_bytes_[name] = bytes;
        LayerGroup& g = groups_[idx]; // creates on demand
        g.layer_index = idx;
        g.tensor_names.push_back(name);
        g.bytes += bytes;
        total_model_bytes_ += bytes;
    }

    for (const auto& kv : groups_) {
        if (kv.first >= 0) layer_order_.push_back(kv.first);
    }
    std::sort(layer_order_.begin(), layer_order_.end());

    auto shared_it = groups_.find(-1);
    shared_bytes_ = (shared_it != groups_.end()) ? shared_it->second.bytes : 0;

    // Shared tensors are always resident; advise them in.
    if (shared_it != groups_.end()) {
        for (const std::string& name : shared_it->second.tensor_names) {
            advise_range(loader_.get_tensor_raw_data(name), tensor_bytes_[name], MADV_WILLNEED);
        }
    }
    current_resident_bytes_ = shared_bytes_;
    peak_resident_bytes_ = current_resident_bytes_;

    is_open_ = true;
    return true;
}

void GGUFLayerStreamer::close() {
    loader_.close();
    groups_.clear();
    layer_order_.clear();
    tensor_bytes_.clear();
    resident_lru_.clear();
    resident_set_.clear();
    current_resident_bytes_ = 0;
    peak_resident_bytes_ = 0;
    total_model_bytes_ = 0;
    shared_bytes_ = 0;
    is_open_ = false;
}

std::vector<std::string> GGUFLayerStreamer::layer_tensor_names(int layer_index) const {
    auto it = groups_.find(layer_index);
    if (it == groups_.end()) return {};
    return it->second.tensor_names;
}

const uint8_t* GGUFLayerStreamer::tensor_data(const std::string& name) const {
    return loader_.get_tensor_raw_data(name);
}

bool GGUFLayerStreamer::acquire_layer(int layer_index) {
    auto it = groups_.find(layer_index);
    if (it == groups_.end() || layer_index < 0) return false;

    if (resident_set_.count(layer_index)) {
        // Refresh LRU position.
        resident_lru_.erase(
            std::remove(resident_lru_.begin(), resident_lru_.end(), layer_index),
            resident_lru_.end());
        resident_lru_.push_back(layer_index);
    } else {
        // Evict LRU first so the budget is a hard bound (no transient
        // double-residency on insert; prefetch stays advisory-only).
        while (resident_lru_.size() >= resident_layer_count_) {
            int victim = resident_lru_.front();
            resident_lru_.pop_front();
            resident_set_.erase(victim);
            auto vit = groups_.find(victim);
            if (vit == groups_.end()) continue;
            current_resident_bytes_ -= vit->second.bytes;
            for (const std::string& name : vit->second.tensor_names) {
                advise_range(loader_.get_tensor_raw_data(name), tensor_bytes_[name], MADV_DONTNEED);
            }
        }
        resident_set_.insert(layer_index);
        resident_lru_.push_back(layer_index);
        current_resident_bytes_ += it->second.bytes;
        if (current_resident_bytes_ > peak_resident_bytes_) {
            peak_resident_bytes_ = current_resident_bytes_;
        }
        // Fault pages in (advisory; reads fault them for real).
        for (const std::string& name : it->second.tensor_names) {
            advise_range(loader_.get_tensor_raw_data(name), tensor_bytes_[name], MADV_WILLNEED);
        }
    }

    // Prefetch next layer (advisory only, not counted as resident).
    auto pos = std::find(layer_order_.begin(), layer_order_.end(), layer_index);
    if (pos != layer_order_.end() && (pos + 1) != layer_order_.end()) {
        int next = *(pos + 1);
        auto nit = groups_.find(next);
        if (nit != groups_.end() && !resident_set_.count(next)) {
            for (const std::string& name : nit->second.tensor_names) {
                advise_range(loader_.get_tensor_raw_data(name), tensor_bytes_[name], MADV_WILLNEED);
            }
        }
    }

    evict_if_over_budget();
    return true;
}

void GGUFLayerStreamer::evict_if_over_budget() {
    while (resident_lru_.size() > resident_layer_count_) {
        int victim = resident_lru_.front();
        resident_lru_.pop_front();
        resident_set_.erase(victim);
        auto it = groups_.find(victim);
        if (it == groups_.end()) continue;
        current_resident_bytes_ -= it->second.bytes;
        for (const std::string& name : it->second.tensor_names) {
            advise_range(loader_.get_tensor_raw_data(name), tensor_bytes_[name], MADV_DONTNEED);
        }
    }
}

} // namespace metal_llm
