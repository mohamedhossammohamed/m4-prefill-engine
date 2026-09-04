#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

// PrismML Q2_0: 128 weights per block
// 2 bytes: FP16 scale (shared across 128 weights)
// 32 bytes: packed 2-bit codes, 4 codes per byte, LSB-first
// Total: 34 bytes per block (2.125 bits/weight)
struct block_prism_q2_0 {
    half d;
    uint8_t qs[32];
};

struct CodecPrismQ2_0 {
    using BlockType = block_prism_q2_0;
    enum { BLOCK_SIZE = 128, SUPER_BLOCK_SIZE = 128 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    // The GEMM core iterates kb in slices of 32 weights (kb in 0 .. K/32 - 1).
    // Since each Prism Q2_0 block contains 128 weights, 1 block = 4 sub-blocks of 32:
    //   blk_idx = kb / 4
    //   sub_idx = kb % 4
    // Each 32-weight sub-block maps to 8 bytes in qs (8 * 4 = 32 codes).
    static inline void unpack_column(
        device const block_prism_q2_0* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint n_blocks_k = K / 128;
        uint blk_idx = kb / 4;
        uint sub_idx = kb % 4;

        block_prism_q2_0 blk = B[col * n_blocks_k + blk_idx];
        half d = blk.d;

        uint qs_offset = sub_idx * 8;
        uint32_t q0 = read_u32_unaligned(blk.qs + qs_offset + 0);
        uint32_t q1 = read_u32_unaligned(blk.qs + qs_offset + 4);

        // Dequantization formula: w = (q - 1) * scale
        // q=0 -> -1 * scale
        // q=1 ->  0 * scale
        // q=2 -> +1 * scale
        // q=3 -> +2 * scale (matching PrismML's published mathematical specification;
        //        PrismML guarantees q=3 is never emitted in valid ternary weights, but
        //        per directive, it evaluates to (3 - 1) * scale = +2 * scale and is
        //        NOT silently clamped to 0).
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int c0 = (int)((q0 >> (j * 2)) & 0x3) - 1;
            sh_B[j * 64 + linear_tid] = (half)((float)c0 * (float)d);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int c1 = (int)((q1 >> (j * 2)) & 0x3) - 1;
            sh_B[(16 + j) * 64 + linear_tid] = (half)((float)c1 * (float)d);
        }
    }
};

} // namespace quant
} // namespace metal_llm
