#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "codec_traits.metal"

namespace metal_llm {
namespace quant {

struct block_mlx_4bit {
    half d;
    half bias;
    uint8_t qs[16];
};

struct CodecMLX4Bit {
    using BlockType = block_mlx_4bit;
    enum { BLOCK_SIZE = 32, SUPER_BLOCK_SIZE = 32 };

    // Unpacks 32 weights corresponding to (col, kb) into sh_B[0..31][linear_tid]
    static inline void unpack_column(
        device const block_mlx_4bit* B,
        uint col,
        uint kb,
        uint K,
        threadgroup half* sh_B,
        uint linear_tid)
    {
        uint nb = K / 32;
        block_mlx_4bit blk = B[col * nb + kb];

        half d = blk.d;
        half bias = blk.bias;
        half4 hd = half4(d);
        half4 h_bias = half4(bias);

        uint w0 = read_u32_unaligned(blk.qs + 0);
        uint w1 = read_u32_unaligned(blk.qs + 4);
        uint w2 = read_u32_unaligned(blk.qs + 8);
        uint w3 = read_u32_unaligned(blk.qs + 12);

        half4 vl[4], vh[4];
        vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_bias);
        vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_bias);
        vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_bias);
        vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_bias);
        vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_bias);
        vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_bias);
        vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_bias);
        vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_bias);

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
