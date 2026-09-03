#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include "core/memory/quant_types.h"

namespace metal_llm {

using namespace core::memory;

struct QuantCodecDescriptor {
    QuantFormat format;
    std::string name;
    std::string description;
    uint32_t block_size;
    size_t struct_size;
    double bits_per_weight;
    std::string gemm_kernel_name;
    std::string head_gemm_kernel_name;
    std::function<void(const void* src, void* dst, size_t count)> quantize_fn;
    std::function<void(const void* src, float* dst, size_t count)> dequantize_fn;
};

class QuantRegistry {
public:
    static QuantRegistry& instance();

    // Register a new codec descriptor
    void register_codec(const QuantCodecDescriptor& desc);

    // Retrieve descriptor by format enum
    const QuantCodecDescriptor* get(QuantFormat format) const;

    // Retrieve descriptor by name string
    const QuantCodecDescriptor* get_by_name(const std::string& name) const;

    // List all registered formats
    std::vector<QuantFormat> available_formats() const;

    // Compute required memory allocation size in bytes
    size_t compute_weight_bytes(QuantFormat format, size_t num_elements) const;

    // Check if format is registered
    bool has_format(QuantFormat format) const;

private:
    QuantRegistry();
    void register_builtin_codecs();

    std::map<QuantFormat, QuantCodecDescriptor> registry_;
    std::map<std::string, QuantFormat> name_to_format_;
};

// Auto-registration helper class
struct CodecRegistrationHelper {
    CodecRegistrationHelper(const QuantCodecDescriptor& desc) {
        QuantRegistry::instance().register_codec(desc);
    }
};

#define REGISTER_QUANT_CODEC(...) \
    static ::metal_llm::CodecRegistrationHelper _codec_reg_##__LINE__(__VA_ARGS__);

} // namespace metal_llm
