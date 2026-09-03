#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

struct block_var_rate_affine {
    half d;
    half bias;
    uint8_t scales[8];
    uint8_t biases[8];
    uint8_t modes[8];
    uint8_t _pad[4];
    uint8_t qs[128];
};

struct CodecVarRateAffine {
    using BlockType = block_var_rate_affine;
    enum { BLOCK_SIZE = 32, SUPER_BLOCK_SIZE = 256 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    static inline void unpack_column(
        device const block_var_rate_affine* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint n_super = K / 256;
        uint sb = kb / 8;
        uint sub_idx = kb % 8;

        block_var_rate_affine blk = B[col * n_super + sb];

        half d = blk.d;
        half bias = blk.bias;
        uint8_t sc_raw = blk.scales[sub_idx];
        uint8_t bi_raw = blk.biases[sub_idx];
        uint8_t mode = blk.modes[sub_idx];

        half sub_scale = d * (half)sc_raw * 0.0625h;
        half sub_bias  = bias + d * ((half)bi_raw - 128.0h) * 0.0625h;

        uint bit_depth = mode & 0x07;
        uint perm_mode = (mode >> 3) & 0x03;

        constexpr uint sub_offsets[8] = {0, 12, 24, 40, 56, 72, 88, 108};
        uint offset = sub_offsets[sub_idx];
        thread const uint8_t* p = blk.qs + offset;

        half unpacked_w[32];

        if (bit_depth == 3) {
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

                unpacked_w[g * 8 + 0] = fma((half)((int)q0 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 1] = fma((half)((int)q1 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 2] = fma((half)((int)q2 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 3] = fma((half)((int)q3 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 4] = fma((half)((int)q4 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 5] = fma((half)((int)q5 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 6] = fma((half)((int)q6 - 4), sub_scale, sub_bias);
                unpacked_w[g * 8 + 7] = fma((half)((int)q7 - 4), sub_scale, sub_bias);
            }
        } else if (bit_depth == 5) {
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                uint8_t b0 = p[g * 5 + 0];
                uint8_t b1 = p[g * 5 + 1];
                uint8_t b2 = p[g * 5 + 2];
                uint8_t b3 = p[g * 5 + 3];
                uint8_t b4 = p[g * 5 + 4];

                uint8_t q0 = (b0) & 0x1F;
                uint8_t q1 = ((b0 >> 5) | (b1 << 3)) & 0x1F;
                uint8_t q2 = (b1 >> 2) & 0x1F;
                uint8_t q3 = ((b1 >> 7) | (b2 << 1)) & 0x1F;
                uint8_t q4 = ((b2 >> 4) | (b3 << 4)) & 0x1F;
                uint8_t q5 = (b3 >> 1) & 0x1F;
                uint8_t q6 = ((b3 >> 6) | (b4 << 2)) & 0x1F;
                uint8_t q7 = (b4 >> 3) & 0x1F;

                unpacked_w[g * 8 + 0] = fma((half)((int)q0 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 1] = fma((half)((int)q1 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 2] = fma((half)((int)q2 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 3] = fma((half)((int)q3 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 4] = fma((half)((int)q4 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 5] = fma((half)((int)q5 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 6] = fma((half)((int)q6 - 16), sub_scale, sub_bias);
                unpacked_w[g * 8 + 7] = fma((half)((int)q7 - 16), sub_scale, sub_bias);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                uint8_t byte_val = p[i];
                uint8_t q0 = byte_val & 0x0F;
                uint8_t q1 = byte_val >> 4;
                unpacked_w[i]      = fma((half)((int)q0 - 8), sub_scale, sub_bias);
                unpacked_w[16 + i] = fma((half)((int)q1 - 8), sub_scale, sub_bias);
            }
        }

        #pragma unroll
        for (int j = 0; j < 32; j++) {
            uint dest_idx = j;
            if (perm_mode == 1) {
                dest_idx = ((j & 1) << 4) | (j >> 1);
            } else if (perm_mode == 2) {
                dest_idx = 31 - j;
            }
            sh_B[dest_idx * 64 + linear_tid] = unpacked_w[j];
        }
    }
};

} // namespace quant
} // namespace metal_llm
