#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"

namespace metal_llm {
namespace quant {

// Quantization format traits
enum QuantFormatType {
    FORMAT_Q4_0 = 0,
    FORMAT_MLX_4BIT = 1,
    FORMAT_Q4_K = 2,
    FORMAT_TERNARY_1_58 = 3,
    FORMAT_VAR_RATE_AFFINE = 4,
    FORMAT_EXL3 = 5
};

} // namespace quant
} // namespace metal_llm
