#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct CodecQ4_K {
    using BlockType = block_q4_K;
    enum { BLOCK_SIZE = 32, SUPER_BLOCK_SIZE = 256 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    static inline void unpack_column(
        device const block_q4_K* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint n_super = K / 256;
        uint sb = kb / 8;
        uint sub_idx = kb % 8;

        block_q4_K blk = B[col * n_super + sb];

        half d = blk.d;
        half dmin = blk.dmin;

        uint8_t sc_raw, min_raw;
        if (sub_idx < 4) {
            sc_raw = blk.scales[sub_idx] & 0x3F;
            min_raw = (blk.scales[sub_idx + 8] & 0x0F) | ((blk.scales[sub_idx] >> 6) << 4);
        } else {
            sc_raw = blk.scales[sub_idx] & 0x3F;
            min_raw = ((blk.scales[sub_idx + 4] >> 4) & 0x0F) | ((blk.scales[sub_idx] >> 6) << 4);
        }

        half d_sub = d * (half)sc_raw;
        half m_sub = dmin * (half)min_raw;
        half4 hd = half4(d_sub);
        half4 h_sub = half4(-m_sub);

        uint qs_offset = sub_idx * 16;
        uint w0 = read_u32_unaligned(blk.qs + qs_offset + 0);
        uint w1 = read_u32_unaligned(blk.qs + qs_offset + 4);
        uint w2 = read_u32_unaligned(blk.qs + qs_offset + 8);
        uint w3 = read_u32_unaligned(blk.qs + qs_offset + 12);

        half4 vl[4], vh[4];
        vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_sub);
        vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_sub);
        vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_sub);
        vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_sub);
        vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_sub);
        vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_sub);
        vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_sub);
        vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_sub);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sh_B[(i * 4 + 0) * 64 + linear_tid] = vl[i][0];
            sh_B[(i * 4 + 1) * 64 + linear_tid] = vl[i][1];
            sh_B[(i * 4 + 2) * 64 + linear_tid] = vl[i][2];
            sh_B[(i * 4 + 3) * 64 + linear_tid] = vl[i][3];
            sh_B[(16 + i * 4 + 0) * 64 + linear_tid] = vh[i][0];
            sh_B[(16 + i * 4 + 1) * 64 + linear_tid] = vh[i][1];
            sh_B[(16 + i * 4 + 2) * 64 + linear_tid] = vh[i][2];
            sh_B[(16 + i * 4 + 3) * 64 + linear_tid] = vh[i][3];
        }
    }
};

} // namespace quant
} // namespace metal_llm
