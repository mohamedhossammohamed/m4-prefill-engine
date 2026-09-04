#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <cstddef>
#include "core/memory/quant_types.h"

namespace metal_llm {

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct GGUFMetadataValue {
    GGUFValueType type = GGUFValueType::UINT32;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string str_val;
    std::vector<GGUFMetadataValue> arr_val;
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0; // Byte offset relative to tensor_data_offset
    size_t size_bytes = 0;
};

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    // Disable copy semantics
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    // Open and parse GGUF file from disk (via zero-copy mmap)
    bool open(const std::string& filepath);

    // Parse from in-memory byte buffer (for synthetic fixtures & tests)
    bool parse_from_buffer(const uint8_t* data, size_t size);

    void close();

    bool is_open() const { return is_open_; }
    uint32_t version() const { return version_; }
    uint64_t tensor_count() const { return n_tensors_; }
    uint64_t metadata_count() const { return n_kv_; }
    uint32_t alignment() const { return alignment_; }
    uint64_t tensor_data_offset() const { return tensor_data_offset_; }
    size_t file_size() const { return file_size_; }

    // Metadata access
    bool has_metadata(const std::string& key) const;
    const GGUFMetadataValue* get_metadata(const std::string& key) const;
    std::string get_string_metadata(const std::string& key, const std::string& def = "") const;
    uint32_t get_uint32_metadata(const std::string& key, uint32_t def = 0) const;
    uint64_t get_uint64_metadata(const std::string& key, uint64_t def = 0) const;

    // Tensor access
    bool has_tensor(const std::string& name) const;
    const GGUFTensorInfo* get_tensor_info(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    // Direct pointer to raw tensor data in file/memory
    const uint8_t* get_tensor_raw_data(const std::string& name) const;

    // Endian-safe extraction of PrismML Q2_0 tensor into block_prism_q2_0 array
    // Validates:
    // 1. Tensor existence and non-zero dimensions
    // 2. Total element count divisibility by 128
    // 3. File bounds / buffer size
    // Unpacks:
    // - Little-endian FP16 scale into block_prism_q2_0::d
    // - 32-byte packed 2-bit codes into block_prism_q2_0::qs
    bool extract_q2_0_tensor(
        const std::string& name,
        std::vector<core::memory::block_prism_q2_0>& out_blocks,
        uint64_t* out_k = nullptr,
        uint64_t* out_n = nullptr) const;

    // Endian-safe read helpers (public for testing and fixtures)
    static uint16_t read_le16(const uint8_t* p);
    static uint32_t read_le32(const uint8_t* p);
    static uint64_t read_le64(const uint8_t* p);
    static float read_le_f32(const uint8_t* p);
    static double read_le_f64(const uint8_t* p);

private:
    bool parse_internal(const uint8_t* buf, size_t size);
    bool read_string(const uint8_t* buf, size_t size, size_t& offset, std::string& out_str);
    bool read_value(const uint8_t* buf, size_t size, size_t& offset, GGUFValueType type, GGUFMetadataValue& out_val);

    bool is_open_ = false;
    int fd_ = -1;
    const uint8_t* data_ptr_ = nullptr;
    size_t file_size_ = 0;
    bool is_mmap_ = false;

    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0;
    uint64_t n_kv_ = 0;
    uint32_t alignment_ = 32;
    uint64_t tensor_data_offset_ = 0;

    std::map<std::string, GGUFMetadataValue> metadata_;
    std::map<std::string, GGUFTensorInfo> tensors_;
    std::vector<std::string> tensor_order_;
};

} // namespace metal_llm
