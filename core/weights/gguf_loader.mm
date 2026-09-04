#include "gguf_loader.h"
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace metal_llm {

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

void GGUFLoader::close() {
    if (is_open_) {
        if (is_mmap_ && data_ptr_ && file_size_ > 0) {
            munmap(const_cast<uint8_t*>(data_ptr_), file_size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        data_ptr_ = nullptr;
        file_size_ = 0;
        is_mmap_ = false;
        is_open_ = false;
        metadata_.clear();
        tensors_.clear();
        tensor_order_.clear();
    }
}

uint16_t GGUFLoader::read_le16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

uint32_t GGUFLoader::read_le32(const uint8_t* p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

uint64_t GGUFLoader::read_le64(const uint8_t* p) {
    return (uint64_t)read_le32(p) | ((uint64_t)read_le32(p + 4) << 32);
}

float GGUFLoader::read_le_f32(const uint8_t* p) {
    uint32_t u = read_le32(p);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

double GGUFLoader::read_le_f64(const uint8_t* p) {
    uint64_t u = read_le64(p);
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

bool GGUFLoader::read_string(const uint8_t* buf, size_t size, size_t& offset, std::string& out_str) {
    if (offset + 8 > size) return false;
    uint64_t len = read_le64(buf + offset);
    offset += 8;
    if (offset + len > size) return false;
    out_str.assign(reinterpret_cast<const char*>(buf + offset), len);
    offset += len;
    return true;
}

bool GGUFLoader::read_value(const uint8_t* buf, size_t size, size_t& offset, GGUFValueType type, GGUFMetadataValue& out_val) {
    out_val.type = type;
    switch (type) {
        case GGUFValueType::UINT8:
            if (offset + 1 > size) return false;
            out_val.uint_val = buf[offset++];
            return true;
        case GGUFValueType::INT8:
            if (offset + 1 > size) return false;
            out_val.int_val = (int8_t)buf[offset++];
            return true;
        case GGUFValueType::UINT16:
            if (offset + 2 > size) return false;
            out_val.uint_val = read_le16(buf + offset);
            offset += 2;
            return true;
        case GGUFValueType::INT16:
            if (offset + 2 > size) return false;
            out_val.int_val = (int16_t)read_le16(buf + offset);
            offset += 2;
            return true;
        case GGUFValueType::UINT32:
            if (offset + 4 > size) return false;
            out_val.uint_val = read_le32(buf + offset);
            offset += 4;
            return true;
        case GGUFValueType::INT32:
            if (offset + 4 > size) return false;
            out_val.int_val = (int32_t)read_le32(buf + offset);
            offset += 4;
            return true;
        case GGUFValueType::FLOAT32:
            if (offset + 4 > size) return false;
            out_val.float_val = read_le_f32(buf + offset);
            offset += 4;
            return true;
        case GGUFValueType::BOOL:
            if (offset + 1 > size) return false;
            out_val.uint_val = (buf[offset++] != 0) ? 1 : 0;
            return true;
        case GGUFValueType::STRING:
            return read_string(buf, size, offset, out_val.str_val);
        case GGUFValueType::ARRAY: {
            if (offset + 4 + 8 > size) return false;
            uint32_t item_type_raw = read_le32(buf + offset);
            offset += 4;
            uint64_t count = read_le64(buf + offset);
            offset += 8;
            GGUFValueType item_type = static_cast<GGUFValueType>(item_type_raw);
            out_val.arr_val.resize(count);
            for (uint64_t i = 0; i < count; i++) {
                if (!read_value(buf, size, offset, item_type, out_val.arr_val[i])) {
                    return false;
                }
            }
            return true;
        }
        case GGUFValueType::UINT64:
            if (offset + 8 > size) return false;
            out_val.uint_val = read_le64(buf + offset);
            offset += 8;
            return true;
        case GGUFValueType::INT64:
            if (offset + 8 > size) return false;
            out_val.int_val = (int64_t)read_le64(buf + offset);
            offset += 8;
            return true;
        case GGUFValueType::FLOAT64:
            if (offset + 8 > size) return false;
            out_val.float_val = read_le_f64(buf + offset);
            offset += 8;
            return true;
        default:
            std::cerr << "[GGUFLoader] Unknown metadata value type: " << (uint32_t)type << std::endl;
            return false;
    }
}

bool GGUFLoader::open(const std::string& filepath) {
    close();

    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "[GGUFLoader] Failed to open file: " << filepath << std::endl;
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0 || st.st_size < 24) {
        std::cerr << "[GGUFLoader] Invalid file size: " << filepath << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    file_size_ = (size_t)st.st_size;
    void* mapped = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "[GGUFLoader] Failed to mmap file: " << filepath << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    data_ptr_ = static_cast<const uint8_t*>(mapped);
    is_mmap_ = true;

    if (!parse_internal(data_ptr_, file_size_)) {
        close();
        return false;
    }

    is_open_ = true;
    return true;
}

bool GGUFLoader::parse_from_buffer(const uint8_t* data, size_t size) {
    close();
    if (!data || size < 24) return false;

    data_ptr_ = data;
    file_size_ = size;
    is_mmap_ = false;

    if (!parse_internal(data_ptr_, file_size_)) {
        close();
        return false;
    }

    is_open_ = true;
    return true;
}

bool GGUFLoader::parse_internal(const uint8_t* buf, size_t size) {
    if (size < 24) return false;

    // 1. Magic check: 'G', 'G', 'U', 'F'
    if (buf[0] != 'G' || buf[1] != 'G' || buf[2] != 'U' || buf[3] != 'F') {
        std::cerr << "[GGUFLoader] Invalid magic header (expected 'GGUF')" << std::endl;
        return false;
    }

    // 2. Version check (v2 or v3)
    version_ = read_le32(buf + 4);
    if (version_ < 2 || version_ > 3) {
        std::cerr << "[GGUFLoader] Unsupported GGUF version: " << version_ << std::endl;
        return false;
    }

    // 3. Tensor and KV counts
    n_tensors_ = read_le64(buf + 8);
    n_kv_ = read_le64(buf + 16);
    alignment_ = 32; // Default GGUF alignment

    size_t offset = 24;

    // 4. Parse Metadata KV table
    for (uint64_t i = 0; i < n_kv_; i++) {
        std::string key;
        if (!read_string(buf, size, offset, key)) {
            std::cerr << "[GGUFLoader] Error reading metadata key at index " << i << std::endl;
            return false;
        }

        if (offset + 4 > size) return false;
        uint32_t val_type_raw = read_le32(buf + offset);
        offset += 4;

        GGUFMetadataValue val;
        if (!read_value(buf, size, offset, static_cast<GGUFValueType>(val_type_raw), val)) {
            std::cerr << "[GGUFLoader] Error reading metadata value for key: " << key << std::endl;
            return false;
        }

        // Handle alignment override if present
        if (key == "general.alignment") {
            alignment_ = (uint32_t)val.uint_val;
            if (alignment_ == 0) alignment_ = 32;
        }

        metadata_[key] = val;
    }

    // 5. Parse Tensor Info table
    tensor_order_.clear();
    tensors_.clear();
    tensor_order_.reserve(n_tensors_);

    for (uint64_t i = 0; i < n_tensors_; i++) {
        GGUFTensorInfo info;
        if (!read_string(buf, size, offset, info.name)) {
            std::cerr << "[GGUFLoader] Error reading tensor name at index " << i << std::endl;
            return false;
        }

        if (offset + 4 > size) return false;
        info.n_dims = read_le32(buf + offset);
        offset += 4;

        if (offset + (size_t)info.n_dims * 8 > size) return false;
        info.dims.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; d++) {
            info.dims[d] = read_le64(buf + offset);
            offset += 8;
        }

        if (offset + 4 + 8 > size) return false;
        info.type = read_le32(buf + offset);
        offset += 4;
        info.offset = read_le64(buf + offset);
        offset += 8;

        tensor_order_.push_back(info.name);
        tensors_[info.name] = info;
    }

    // 6. Calculate tensor data offset (aligned to alignment_)
    size_t header_end = offset;
    tensor_data_offset_ = ((header_end + alignment_ - 1) / alignment_) * alignment_;

    if (tensor_data_offset_ > size && n_tensors_ > 0) {
        std::cerr << "[GGUFLoader] File truncated: tensor_data_offset exceeds file size" << std::endl;
        return false;
    }

    return true;
}

bool GGUFLoader::has_metadata(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

const GGUFMetadataValue* GGUFLoader::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? &it->second : nullptr;
}

std::string GGUFLoader::get_string_metadata(const std::string& key, const std::string& def) const {
    const auto* v = get_metadata(key);
    return (v && v->type == GGUFValueType::STRING) ? v->str_val : def;
}

uint32_t GGUFLoader::get_uint32_metadata(const std::string& key, uint32_t def) const {
    const auto* v = get_metadata(key);
    return (v && (v->type == GGUFValueType::UINT32 || v->type == GGUFValueType::UINT64)) ? (uint32_t)v->uint_val : def;
}

uint64_t GGUFLoader::get_uint64_metadata(const std::string& key, uint64_t def) const {
    const auto* v = get_metadata(key);
    return (v && (v->type == GGUFValueType::UINT64 || v->type == GGUFValueType::UINT32)) ? v->uint_val : def;
}

bool GGUFLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const GGUFTensorInfo* GGUFLoader::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

std::vector<std::string> GGUFLoader::tensor_names() const {
    return tensor_order_;
}

const uint8_t* GGUFLoader::get_tensor_raw_data(const std::string& name) const {
    const auto* info = get_tensor_info(name);
    if (!info || !data_ptr_) return nullptr;

    size_t target_offset = tensor_data_offset_ + info->offset;
    if (target_offset >= file_size_) return nullptr;
    return data_ptr_ + target_offset;
}

bool GGUFLoader::extract_q2_0_tensor(
    const std::string& name,
    std::vector<core::memory::block_prism_q2_0>& out_blocks,
    uint64_t* out_k,
    uint64_t* out_n) const
{
    const auto* info = get_tensor_info(name);
    if (!info) {
        std::cerr << "[GGUFLoader] Tensor not found: " << name << std::endl;
        return false;
    }

    if (info->dims.empty()) {
        std::cerr << "[GGUFLoader] Tensor has 0 dimensions: " << name << std::endl;
        return false;
    }

    uint64_t total_elements = 1;
    for (uint64_t dim : info->dims) {
        total_elements *= dim;
    }

    if (total_elements == 0 || (total_elements % 128) != 0) {
        std::cerr << "[GGUFLoader] Total elements (" << total_elements
                  << ") not divisible by 128 for Q2_0 tensor: " << name << std::endl;
        return false;
    }

    uint64_t num_blocks = total_elements / 128;
    size_t required_bytes = num_blocks * sizeof(core::memory::block_prism_q2_0);
    size_t abs_offset = tensor_data_offset_ + info->offset;

    if (abs_offset + required_bytes > file_size_) {
        std::cerr << "[GGUFLoader] File bounds violation for tensor: " << name
                  << " (offset=" << abs_offset << " + " << required_bytes
                  << " > file_size=" << file_size_ << ")" << std::endl;
        return false;
    }

    const uint8_t* raw_src = data_ptr_ + abs_offset;
    out_blocks.resize(num_blocks);

    // Endianness-safe unpack:
    // 2 bytes: little-endian FP16 scale -> block_prism_q2_0::d
    // 32 bytes: packed 2-bit codes -> block_prism_q2_0::qs
    for (uint64_t b = 0; b < num_blocks; b++) {
        const uint8_t* blk_ptr = raw_src + b * 34;
        uint16_t scale_le = read_le16(blk_ptr);
        std::memcpy(&out_blocks[b].d, &scale_le, sizeof(QUANT_HALF));
        std::memcpy(out_blocks[b].qs, blk_ptr + 2, 32);
    }

    // In GGUF, 2D tensors are [columns, rows] = [K, N]
    if (out_k) {
        *out_k = info->dims[0];
    }
    if (out_n) {
        *out_n = (info->dims.size() >= 2) ? info->dims[1] : 1;
    }

    return true;
}

} // namespace metal_llm
