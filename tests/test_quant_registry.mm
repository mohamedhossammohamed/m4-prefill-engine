#import "src/router/quant_registry.h"
#include <iostream>
#include <cassert>

using namespace metal_llm;

int main() {
    std::cout << "[*] Testing QuantRegistry (All 8 Formats + Dynamic Extensibility)..." << std::endl;

    auto& reg = QuantRegistry::instance();
    auto formats = reg.available_formats();
    assert(formats.size() >= 8);
    std::cout << "    -> Available formats count: " << formats.size() << " (PASSED)" << std::endl;

    // 1. QUANT_Q4_0
    {
        const auto* desc = reg.get(QUANT_Q4_0);
        assert(desc != nullptr && desc->name == "Q4_0");
        assert(desc->block_size == 32 && desc->struct_size == 18);
        assert(desc->bits_per_weight == 4.50);
        assert(desc->gemm_kernel_name == "quant_router_gemm_q4_0_64x64");
        assert(reg.compute_weight_bytes(QUANT_Q4_0, 4096 * 4096) == (4096 * 4096 / 32) * 18);
        std::cout << "    -> [1/8] QUANT_Q4_0 descriptor verified." << std::endl;
    }

    // 2. QUANT_MLX_4BIT
    {
        const auto* desc = reg.get(QUANT_MLX_4BIT);
        assert(desc != nullptr && desc->name == "MLX_4BIT");
        assert(desc->block_size == 32 && desc->struct_size == 20);
        assert(desc->bits_per_weight == 5.00);
        assert(desc->gemm_kernel_name == "quant_router_gemm_mlx_4bit_64x64");
        assert(reg.compute_weight_bytes(QUANT_MLX_4BIT, 4096 * 4096) == (4096 * 4096 / 32) * 20);
        std::cout << "    -> [2/8] QUANT_MLX_4BIT descriptor verified." << std::endl;
    }

    // 3. QUANT_Q4_K
    {
        const auto* desc = reg.get(QUANT_Q4_K);
        assert(desc != nullptr && desc->name == "Q4_K");
        assert(desc->block_size == 256 && desc->struct_size == 144);
        assert(desc->bits_per_weight == 4.50);
        assert(desc->gemm_kernel_name == "quant_router_gemm_q4_k_64x64");
        assert(reg.compute_weight_bytes(QUANT_Q4_K, 4096 * 4096) == (4096 * 4096 / 256) * 144);
        std::cout << "    -> [3/8] QUANT_Q4_K descriptor verified." << std::endl;
    }

    // 4. QUANT_TERNARY_1_58
    {
        const auto* desc = reg.get(QUANT_TERNARY_1_58);
        assert(desc != nullptr && desc->name == "TERNARY_1_58");
        assert(desc->block_size == 32 && desc->struct_size == 12);
        assert(desc->bits_per_weight == 3.00);
        assert(desc->gemm_kernel_name == "quant_router_gemm_ternary_1_58_64x64");
        assert(reg.compute_weight_bytes(QUANT_TERNARY_1_58, 4096 * 4096) == (4096 * 4096 / 32) * 12);
        std::cout << "    -> [4/8] QUANT_TERNARY_1_58 descriptor verified." << std::endl;
    }

    // 5. QUANT_VAR_RATE_AFFINE
    {
        const auto* desc = reg.get(QUANT_VAR_RATE_AFFINE);
        assert(desc != nullptr && desc->name == "VAR_RATE_AFFINE");
        assert(desc->block_size == 256 && desc->struct_size == 160);
        assert(desc->bits_per_weight == 5.00);
        assert(desc->gemm_kernel_name == "quant_router_gemm_var_rate_affine_64x64");
        assert(reg.compute_weight_bytes(QUANT_VAR_RATE_AFFINE, 4096 * 4096) == (4096 * 4096 / 256) * 160);
        std::cout << "    -> [5/8] QUANT_VAR_RATE_AFFINE descriptor verified." << std::endl;
    }

    // 6. QUANT_EXL3
    {
        const auto* desc = reg.get(QUANT_EXL3);
        assert(desc != nullptr && desc->name == "EXL3");
        assert(desc->block_size == 256 && desc->struct_size == 144);
        assert(desc->bits_per_weight == 4.50);
        assert(desc->gemm_kernel_name == "quant_router_gemm_exl3_64x64");
        assert(reg.compute_weight_bytes(QUANT_EXL3, 4096 * 4096) == (4096 * 4096 / 256) * 144);
        std::cout << "    -> [6/8] QUANT_EXL3 descriptor verified." << std::endl;
    }

    // 7. QUANT_Q8_0
    {
        const auto* desc = reg.get(QUANT_Q8_0);
        assert(desc != nullptr && desc->name == "Q8_0");
        assert(desc->block_size == 32 && desc->struct_size == 34);
        assert(desc->bits_per_weight == 8.50);
        assert(desc->gemm_kernel_name == "quant_router_gemm_q8_0_64x64");
        assert(reg.compute_weight_bytes(QUANT_Q8_0, 4096 * 4096) == (4096 * 4096 / 32) * 34);
        std::cout << "    -> [7/8] QUANT_Q8_0 descriptor verified." << std::endl;
    }

    // 8. QUANT_PRISM_Q2_0
    {
        const auto* desc = reg.get(QUANT_PRISM_Q2_0);
        assert(desc != nullptr && desc->name == "PRISM_Q2_0");
        assert(desc->block_size == 128 && desc->struct_size == 34);
        assert(desc->bits_per_weight == 2.125);
        assert(desc->gemm_kernel_name == "quant_router_gemm_prism_q2_0_64x64");
        assert(reg.compute_weight_bytes(QUANT_PRISM_Q2_0, 4096 * 4096) == (4096 * 4096 / 128) * 34);
        std::cout << "    -> [8/8] QUANT_PRISM_Q2_0 descriptor verified." << std::endl;
    }

    // Test dynamic registration
    QuantCodecDescriptor custom_codec = {
        QUANT_CUSTOM,
        "CUSTOM_FP4",
        "Experimental FP4 format",
        32,
        16,
        4.0,
        "custom_gemm",
        "custom_head_gemm",
        nullptr,
        nullptr
    };
    reg.register_codec(custom_codec);
    assert(reg.has_format(QUANT_CUSTOM));
    const auto* custom = reg.get_by_name("CUSTOM_FP4");
    assert(custom != nullptr);
    assert(custom->struct_size == 16);
    std::cout << "    -> Dynamic format registration verified." << std::endl;

    std::cout << "ALL 8 QUANT REGISTRY DESCRIPTORS AND DYNAMIC REGISTRATION VERIFIED!" << std::endl;
    return 0;
}
