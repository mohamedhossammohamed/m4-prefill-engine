#pragma once

#include <metal_stdlib>

// Common infrastructure
#include "common/types.metal"
#include "common/lsu.metal"
#include "common/simd_reduce.metal"
#include "common/sram_tile.metal"
#include "common/math.metal"

// Quantization codecs
#include "quant/codec_traits.metal"
#include "quant/q4_0.metal"
#include "quant/mlx_4bit.metal"
#include "quant/q4_k.metal"
#include "quant/ternary_1_58.metal"
#include "quant/var_rate_affine.metal"
#include "quant/exl3.metal"

// Pluggable ops
#include "ops/gemm_mma.metal"
#include "ops/gemm_ternary_vec.metal"
#include "ops/swiglu_dual_simd.metal"
#include "ops/flash_attention.metal"
