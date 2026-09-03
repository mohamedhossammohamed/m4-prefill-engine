#import "quant_registry.h"
#include <iostream>
#include <cassert>

namespace metal_llm {

QuantRegistry& QuantRegistry::instance() {
    static QuantRegistry reg;
    return reg;
}

QuantRegistry::QuantRegistry() {
    register_builtin_codecs();
}

void QuantRegistry::register_codec(const QuantCodecDescriptor& desc) {
    registry_[desc.format] = desc;
    name_to_format_[desc.name] = desc.format;
}

const QuantCodecDescriptor* QuantRegistry::get(QuantFormat format) const {
    auto it = registry_.find(format);
    if (it != registry_.end()) {
        return &it->second;
    }
    return nullptr;
}

const QuantCodecDescriptor* QuantRegistry::get_by_name(const std::string& name) const {
    auto it = name_to_format_.find(name);
    if (it != name_to_format_.end()) {
        return get(it->second);
    }
    return nullptr;
}

std::vector<QuantFormat> QuantRegistry::available_formats() const {
    std::vector<QuantFormat> formats;
    formats.reserve(registry_.size());
    for (const auto& kv : registry_) {
        formats.push_back(kv.first);
    }
    return formats;
}

bool QuantRegistry::has_format(QuantFormat format) const {
    return registry_.find(format) != registry_.end();
}

size_t QuantRegistry::compute_weight_bytes(QuantFormat format, size_t num_elements) const {
    const auto* desc = get(format);
    if (!desc) {
        std::cerr << "[QuantRegistry] Unknown format: " << (int)format << std::endl;
        return 0;
    }
    size_t num_blocks = (num_elements + desc->block_size - 1) / desc->block_size;
    return num_blocks * desc->struct_size;
}

void QuantRegistry::register_builtin_codecs() {
    // 1. QUANT_Q4_0
    register_codec({
        QUANT_Q4_0,
        "Q4_0",
        "Standard GGUF Q4_0 Symmetric 32-element blocks",
        32,
        sizeof(block_q4_0),
        4.50,
        "quant_router_gemm_q4_0_64x64",
        "quant_router_head_gemm_q4_0_64x64",
        nullptr,
        nullptr
    });

    // 2. QUANT_MLX_4BIT
    register_codec({
        QUANT_MLX_4BIT,
        "MLX_4BIT",
        "MLX-style 4-bit affine with grouped scale and bias",
        32,
        sizeof(block_mlx_4bit),
        5.00,
        "quant_router_gemm_mlx_4bit_64x64",
        "quant_router_head_gemm_mlx_4bit_64x64",
        nullptr,
        nullptr
    });

    // 3. QUANT_Q4_K
    register_codec({
        QUANT_Q4_K,
        "Q4_K",
        "GGUF Q4_K super-block with 8x32 sub-blocks",
        256,
        sizeof(block_q4_K),
        4.50,
        "quant_router_gemm_q4_k_64x64",
        "quant_router_head_gemm_q4_k_64x64",
        nullptr,
        nullptr
    });

    // 4. QUANT_TERNARY_1_58
    register_codec({
        QUANT_TERNARY_1_58,
        "TERNARY_1_58",
        "BitNet 1.58-bit ternary {-1, 0, +1} packed weights",
        32,
        sizeof(block_ternary_1_58),
        3.00,
        "quant_router_gemm_ternary_1_58_64x64",
        "quant_router_head_gemm_ternary_1_58_64x64",
        nullptr,
        nullptr
    });

    // 5. QUANT_VAR_RATE_AFFINE
    register_codec({
        QUANT_VAR_RATE_AFFINE,
        "VAR_RATE_AFFINE",
        "Grouped Variable-Rate Affine 256-elem mixed 3/4/5-bit super-block",
        256,
        sizeof(block_var_rate_affine),
        5.00,
        "quant_router_gemm_var_rate_affine_64x64",
        "quant_router_head_gemm_var_rate_affine_64x64",
        nullptr,
        nullptr
    });

    // 6. QUANT_EXL3
    register_codec({
        QUANT_EXL3,
        "EXL3",
        "ExLlamaV3 256-elem super-block with vector codebook",
        256,
        sizeof(block_exl3),
        4.50,
        "quant_router_gemm_exl3_64x64",
        "quant_router_head_gemm_exl3_64x64",
        nullptr,
        nullptr
    });

    // 7. QUANT_Q8_0
    register_codec({
        QUANT_Q8_0,
        "Q8_0",
        "Standard GGUF Q8_0 Symmetric 32-element blocks (for KV cache)",
        32,
        sizeof(block_q8_0),
        8.50,
        "quant_router_gemm_q8_0_64x64",
        "quant_router_head_gemm_q8_0_64x64",
        nullptr,
        nullptr
    });
}

} // namespace metal_llm
