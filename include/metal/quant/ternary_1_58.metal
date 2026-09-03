#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

struct block_ternary_1_58 {
    half d;
    half _pad;
    uint32_t qs[2];
};

struct CodecTernary158 {
    using BlockType = block_ternary_1_58;
    enum { BLOCK_SIZE = 32, SUPER_BLOCK_SIZE = 32 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    static inline void unpack_column(
        device const block_ternary_1_58* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint nb = K / 32;
        block_ternary_1_58 blk = B[col * nb + kb];

        half d = blk.d;
        uint32_t q0 = blk.qs[0];
        uint32_t q1 = blk.qs[1];

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int c0 = (int)((q0 >> (j * 2)) & 0x3) - 1;
            if (c0 > 1 || c0 < -1) c0 = 0;
            sh_B[j * 64 + linear_tid] = (half)((float)c0 * (float)d);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int c1 = (int)((q1 >> (j * 2)) & 0x3) - 1;
            if (c1 > 1 || c1 < -1) c1 = 0;
            sh_B[(16 + j) * 64 + linear_tid] = (half)((float)c1 * (float)d);
        }
    }
};

} // namespace quant
} // namespace metal_llm
