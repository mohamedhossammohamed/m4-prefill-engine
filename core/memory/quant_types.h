#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __METAL_VERSION__
#define QUANT_HALF half
#else
#define QUANT_HALF __fp16
#endif

namespace core::memory {

// ============================================================================
// UNIVERSAL QUANTIZATION FORMAT IDENTIFIERS
// ============================================================================
enum QuantFormat {
    QUANT_Q4_0            = 0,  // Standard 32-element symmetric block (1 FP16 scale + 16 uint8 bytes = 18 bytes / block)
    QUANT_MLX_4BIT        = 1,  // MLX-style 32-element affine block (1 FP16 scale + 1 FP16 bias + 16 uint8 bytes = 20 bytes / block)
    QUANT_Q4_K            = 2,  // GGUF-style 256-element super-block (2 FP16 + 12 uint8 scales/mins + 128 uint8 qs = 144 bytes / super-block)
    QUANT_TERNARY_1_58    = 3,  // BitNet-style 32-element ternary block (1 FP16 scale + 2 uint32s = 12 bytes / block)
    QUANT_VAR_RATE_AFFINE = 4,  // Grouped Variable-Rate Affine 256-element super-block (grouped scales/biases + 3/4/5-bit streams = 160 bytes)
    QUANT_EXL3            = 5,  // ExLlamaV3 256-element vector codebook super-block (hierarchical centroids + residuals = 144 bytes)
    QUANT_Q8_0            = 6,  // Standard 32-element 8-bit symmetric block (1 FP16 scale + 32 int8 bytes = 34 bytes / block)
    QUANT_PRISM_Q2_0      = 7,  // PrismML 128-element ternary block (1 FP16 scale + 32 packed 2-bit bytes = 34 bytes / block, 2.125 bpw)
    QUANT_CUSTOM          = 99  // Extensible custom format identifier
};

enum LayoutMode {
    LAYOUT_STANDARD    = 0,  // Matrix output [M, N]
    LAYOUT_DIRECT_HEAD = 1   // Direct attention head routing [H, M, D] where N = H * D
};

// ============================================================================
// QUANTIZATION BLOCK DATA STRUCTURES (Exact binary compatibility GPU/CPU)
// ============================================================================

// 1. QUANT_Q4_0: 32 weights per block (18 bytes = 4.50 bits/weight)
struct block_q4_0 {
    QUANT_HALF d;
    uint8_t qs[16];
};

// 2. QUANT_MLX_4BIT: 32 weights per block (20 bytes = 5.00 bits/weight)
// MLX 4-bit affine block with FP16 scale and FP16 bias
struct block_mlx_4bit {
    QUANT_HALF d;     // FP16 scale (2 bytes)
    QUANT_HALF bias;  // FP16 bias (2 bytes)
    uint8_t qs[16];   // 32 x 4-bit nibbles (16 bytes)
};

// 3. QUANT_Q4_K: 256 weights per super-block (144 bytes = 4.50 bits/weight)
struct block_q4_K {
    QUANT_HALF d;
    QUANT_HALF dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

// 4. QUANT_TERNARY_1_58: 32 weights per block (12 bytes = 3.00 bits/weight, 1.58 bits entropy)
struct block_ternary_1_58 {
    QUANT_HALF d;
    QUANT_HALF _pad;
    uint32_t qs[2];
};

// 5. QUANT_VAR_RATE_AFFINE: 256 weights per super-block (160 bytes = 5.00 bits/weight)
struct block_var_rate_affine {
    QUANT_HALF d;          // super-block global scale (2 bytes)
    QUANT_HALF bias;       // super-block global bias (2 bytes)
    uint8_t scales[8];     // 8 x sub-block scales (8 bytes)
    uint8_t biases[8];     // 8 x sub-block biases (8 bytes)
    uint8_t modes[8];      // 8 x sub-block mode & permutation metadata (8 bytes)
    uint8_t _pad[4];       // alignment padding to 32-byte header / 160 bytes total (4 bytes)
    uint8_t qs[128];       // Multi-bit packed quant payload stream (128 bytes)
};

// 6. QUANT_EXL3: 256 weights per super-block (144 bytes = 4.50 bits/weight)
struct block_exl3 {
    QUANT_HALF d;          // super-block global scale (2 bytes)
    QUANT_HALF bias;       // super-block global bias (2 bytes)
    uint8_t scales[8];     // 8 x sub-block scales (8 bytes)
    uint8_t residuals[8];  // 8 x sub-block residual scales (8 bytes)
    int8_t codebook[16];   // 16 x hierarchical vector codebook centroids (16 bytes)
    uint8_t _pad[12];      // alignment padding to 144 bytes (12 bytes)
    uint8_t qs[96];        // 256 x 3-bit packed index streams (96 bytes = 8 x 12 bytes)
};

// 7. QUANT_Q8_0: 32 weights per block (34 bytes = 8.50 bits/weight)
struct block_q8_0 {
    QUANT_HALF d;
    int8_t qs[32];
};

// 8. QUANT_PRISM_Q2_0: 128 weights per block (34 bytes = 2.125 bits/weight)
// 2 bytes FP16 scale + 32 bytes packed 2-bit codes (4 codes per byte, LSB-first)
struct block_prism_q2_0 {
    QUANT_HALF d;
    uint8_t qs[32];
};
#ifndef __METAL_VERSION__
static_assert(sizeof(block_prism_q2_0) == 34, "block_prism_q2_0 must be exactly 34 bytes");
#endif

// ============================================================================
// FORMAT METADATA & HELPER UTILITIES
// ============================================================================
struct QuantFormatInfo {
    QuantFormat format;
    const char* name;
    const char* description;
    uint32_t block_size;        // Elements per block/super-block
    uint32_t block_bytes;       // Bytes per block struct
    double bits_per_weight;     // Theoretical bits per weight
};

inline QuantFormatInfo get_quant_info(QuantFormat fmt) {
    switch (fmt) {
        case QUANT_Q4_0:
            return {QUANT_Q4_0, "Q4_0", "Standard 32-elem symmetric block (FP16 scale)", 32, sizeof(block_q4_0), 4.50};
        case QUANT_MLX_4BIT:
            return {QUANT_MLX_4BIT, "MLX_4BIT", "MLX 32-elem affine block (FP16 scale + FP16 bias)", 32, sizeof(block_mlx_4bit), 5.00};
        case QUANT_Q4_K:
            return {QUANT_Q4_K, "Q4_K", "GGUF 256-elem super-block (6-bit scales & mins)", 256, sizeof(block_q4_K), 4.50};
        case QUANT_TERNARY_1_58:
            return {QUANT_TERNARY_1_58, "TERNARY_1_58", "BitNet 1.58-bit 32-elem ternary {-1, 0, +1} (FP16 scale)", 32, sizeof(block_ternary_1_58), 3.00};
        case QUANT_VAR_RATE_AFFINE:
            return {QUANT_VAR_RATE_AFFINE, "VAR_RATE_AFFINE", "Grouped Variable-Rate Affine 256-elem mixed 3/4/5-bit super-block", 256, sizeof(block_var_rate_affine), 5.00};
        case QUANT_EXL3:
            return {QUANT_EXL3, "EXL3", "ExLlamaV3 256-elem hierarchical vector codebook super-block with residual correction", 256, sizeof(block_exl3), 4.50};
        case QUANT_Q8_0:
            return {QUANT_Q8_0, "Q8_0", "Standard 32-elem 8-bit symmetric block (FP16 scale)", 32, sizeof(block_q8_0), 8.50};
        case QUANT_PRISM_Q2_0:
            return {QUANT_PRISM_Q2_0, "PRISM_Q2_0", "PrismML Q2_0 128-elem ternary {-1, 0, +1} block (FP16 scale)", 128, sizeof(block_prism_q2_0), 2.125};
        case QUANT_CUSTOM:
        default:
            return {QUANT_CUSTOM, "CUSTOM", "Custom / dynamically registered format", 32, 0, 0.0};
    }
}

inline size_t compute_quant_weight_bytes(QuantFormat fmt, size_t total_elements) {
    QuantFormatInfo info = get_quant_info(fmt);
    size_t num_blocks = (total_elements + info.block_size - 1) / info.block_size;
    return num_blocks * info.block_bytes;
}

} // namespace core::memory
