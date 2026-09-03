#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

struct block_exl3 {
    half d;
    half bias;
    uint8_t scales[8];
    uint8_t residuals[8];
    int8_t codebook[16];
    uint8_t _pad[12];
    uint8_t qs[96];
};

struct CodecEXL3 {
    using BlockType = block_exl3;
    enum { BLOCK_SIZE = 32, SUPER_BLOCK_SIZE = 256 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    static inline void unpack_column(
        device const block_exl3* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint n_super = K / 256;
        uint sb = kb / 8;
        uint sub_idx = kb % 8;

        block_exl3 blk = B[col * n_super + sb];

        half d = blk.d;
        half bias = blk.bias;
        half sc_raw = (half)blk.scales[sub_idx];
        half res_raw = (half)blk.residuals[sub_idx];

        half sub_scale = d * sc_raw * 0.015625h;
        half sub_res   = d * (res_raw - 128.0h) * 0.0078125h;

        uint offset = sub_idx * 12;
        thread const uint8_t* p = blk.qs + offset;

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            uint8_t b0 = p[g * 3 + 0];
            uint8_t b1 = p[g * 3 + 1];
            uint8_t b2 = p[g * 3 + 2];

            uint8_t q0 = (b0) & 0x07;
            uint8_t q1 = (b0 >> 3) & 0x07;
            uint8_t q2 = ((b0 >> 6) | (b1 << 2)) & 0x07;
            uint8_t q3 = (b1 >> 1) & 0x07;
            uint8_t q4 = (b1 >> 4) & 0x07;
            uint8_t q5 = ((b1 >> 7) | (b2 << 1)) & 0x07;
            uint8_t q6 = (b2 >> 2) & 0x07;
            uint8_t q7 = (b2 >> 5) & 0x07;

            uint8_t qs[8] = {q0, q1, q2, q3, q4, q5, q6, q7};

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t q = qs[i];
                half c_base = (half)blk.codebook[q];
                half c_res  = (half)blk.codebook[q + 8];
                half w = fma(c_base, sub_scale, fma(c_res, sub_res, bias));
                sh_B[(g * 8 + i) * 64 + linear_tid] = w;
            }
        }
    }
};

} // namespace quant
} // namespace metal_llm
