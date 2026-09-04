#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 1. DATA STRUCTURES & HELPERS
// ============================================================================

// 1. QUANT_Q4_0: Standard 32-element symmetric block (18 bytes = 4.50 bits/weight)
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

// 2. QUANT_MLX_4BIT: MLX 32-element affine block (20 bytes = 5.00 bits/weight)
struct block_mlx_4bit {
    half d;
    half bias;
    uint8_t qs[16];
};

// 3. QUANT_Q4_K: GGUF 256-element super-block (144 bytes = 4.50 bits/weight)
struct block_q4_K {
    half d;              // super-block scale
    half dmin;           // super-block min
    uint8_t scales[12];  // 8 x 6-bit scales + 8 x 6-bit mins
    uint8_t qs[128];     // 128 bytes = 256 x 4-bit nibbles
};

// 4. QUANT_TERNARY_1_58: BitNet 1.58-bit 32-element block (12 bytes = 3.00 bits/weight)
struct block_ternary_1_58 {
    half d;              // shared FP16 scale
    half _pad;           // alignment padding
    uint32_t qs[2];      // 2 x 32-bit = 64 bits = 32 x 2-bit weights in {-1, 0, +1}
};

// 5. QUANT_VAR_RATE_AFFINE: 256 weights per super-block (160 bytes = 5.00 bits/weight)
// Multi-bit rate packing (3-bit, 4-bit, 5-bit sub-block streams) with grouped scale/bias & permutation metadata
struct block_var_rate_affine {
    half d;              // super-block global scale (2 bytes)
    half bias;           // super-block global bias (2 bytes)
    uint8_t scales[8];   // 8 x sub-block scales (8 bytes)
    uint8_t biases[8];   // 8 x sub-block biases (8 bytes)
    uint8_t modes[8];    // 8 x sub-block mode & permutation metadata (8 bytes)
    uint8_t _pad[4];     // alignment padding to 160 bytes (4 bytes)
    uint8_t qs[128];     // Multi-bit packed quant payload stream (128 bytes)
};

// 6. QUANT_EXL3: 256 weights per super-block (144 bytes = 4.50 bits/weight)
// Hierarchical vector codebook centroids/indices, FP16 global scale/bias, sub-block scale & residual metadata
struct block_exl3 {
    half d;              // super-block global scale (2 bytes)
    half bias;           // super-block global bias (2 bytes)
    uint8_t scales[8];   // 8 x sub-block scales (8 bytes)
    uint8_t residuals[8];// 8 x sub-block residual scales (8 bytes)
    int8_t codebook[16]; // 16 x hierarchical vector codebook centroids (16 bytes)
    uint8_t _pad[12];    // alignment padding to 144 bytes (12 bytes)
    uint8_t qs[96];      // 256 x 3-bit packed index streams (96 bytes = 8 x 12 bytes)
};

struct block_prism_q2_0 {
    half d;              // FP16 scale (2 bytes)
    uint8_t qs[32];      // 128 x 2-bit codes, 4 codes per byte, LSB-first (32 bytes)
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// ============================================================================
// 2. HARDWARE CEILING & ROOFLINE PROBES
// ============================================================================
kernel void probe_memory_bandwidth(
    device const float4* src [[buffer(0)]],
    device float4*       dst [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id] + float4(1.0f);
}

kernel void probe_fma_roofline(
    device half* out [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    half4 a0 = half4(1.001h, 1.002h, 1.003h, 1.004h) + half4((half)(id & 0x7) * 0.001h);
    half4 a1 = half4(1.005h, 1.006h, 1.007h, 1.008h);
    half4 b0 = half4(0.999h, 0.998h, 0.997h, 0.996h);
    half4 b1 = half4(0.995h, 0.994h, 0.993h, 0.992h);
    half4 c0 = half4(0.001h);
    half4 c1 = half4(0.002h);

    #pragma unroll(256)
    for (int i = 0; i < 256; i++) {
        c0 = fma(a0, b0, c0);
        c1 = fma(a1, b1, c1);
        a0 = fma(b0, c0, a0);
        a1 = fma(b1, c1, a1);
        b0 = fma(c0, a0, b0);
        b1 = fma(c1, a1, b1);
    }

    if (id < 16) {
        out[id] = c0[0] + c1[0] + a0[0] + a1[0] + b0[0] + b1[0];
    }
}

// ============================================================================
// 3. MODULAR QUANTIZATION UNPACKERS (SRAM STAGING ENGINE)
// ============================================================================

// 3.1 Unpack Q4_0 block (32 weights) into column linear_tid of sh_B[32][64]
inline void unpack_q4_0_block(
    thread const block_q4_0& blk,
    threadgroup half* sh_B,
    uint linear_tid)
{
    half d = blk.d;
    half4 hd = half4(d);
    half4 h_off = half4(-8.0h * d);

    uint w0 = read_u32_unaligned(blk.qs + 0);
    uint w1 = read_u32_unaligned(blk.qs + 4);
    uint w2 = read_u32_unaligned(blk.qs + 8);
    uint w3 = read_u32_unaligned(blk.qs + 12);

    half4 vl[4], vh[4];
    vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
    vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
    vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
    vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
    vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

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

// 3.2 Unpack MLX 4-bit block (32 weights with scale + bias) into sh_B[32][64]
inline void unpack_mlx_4bit_block(
    thread const block_mlx_4bit& blk,
    threadgroup half* sh_B,
    uint linear_tid)
{
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

// 3.3 Unpack Q4_K sub-block (32 weights within 256-weight super-block) into sh_B[32][64]
inline void unpack_q4_K_subblock(
    thread const block_q4_K& blk,
    uint sub_idx, // 0..7
    threadgroup half* sh_B,
    uint linear_tid)
{
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

// 3.4 Unpack Ternary 1.58-bit block (32 weights in {-1, 0, +1}) into sh_B[32][64]
inline void unpack_ternary_1_58_block(
    thread const block_ternary_1_58& blk,
    threadgroup half* sh_B,
    uint linear_tid)
{
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

// 3.5 Unpack Grouped Variable-Rate Affine sub-block (32 weights within 256-weight mixed 3/4/5-bit super-block) into sh_B[32][64]
inline void unpack_var_rate_affine_subblock(
    thread const block_var_rate_affine& blk,
    uint sub_idx, // 0..7
    threadgroup half* sh_B,
    uint linear_tid)
{
    half d = blk.d;
    half bias = blk.bias;
    uint8_t sc_raw = blk.scales[sub_idx];
    uint8_t bi_raw = blk.biases[sub_idx];
    uint8_t mode = blk.modes[sub_idx];

    half sub_scale = d * (half)sc_raw * 0.0625h;
    half sub_bias  = bias + d * ((half)bi_raw - 128.0h) * 0.0625h;

    uint bit_depth = mode & 0x07; // 3, 4, or 5 bits
    uint perm_mode = (mode >> 3) & 0x03; // 0: direct, 1: interleave, 2: reverse

    constexpr uint sub_offsets[8] = {0, 12, 24, 40, 56, 72, 88, 108};
    uint offset = sub_offsets[sub_idx];
    thread const uint8_t* p = blk.qs + offset;

    half unpacked_w[32];

    if (bit_depth == 3) {
        // 12 bytes = 4 groups of 3 bytes (8 values per group)
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
        // 20 bytes = 4 groups of 5 bytes (8 values per group)
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
        // 4-bit default (16 bytes = 32 nibbles)
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

// 3.6 Unpack EXL3 sub-block (32 weights within 256-weight vector codebook super-block) into sh_B[32][64]
inline void unpack_exl3_subblock(
    thread const block_exl3& blk,
    uint sub_idx, // 0..7
    threadgroup half* sh_B,
    uint linear_tid)
{
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

// 3.7 Unpack PrismML Q2_0 sub-block (32 weights within 128-weight block) into sh_B[32][64]
inline void unpack_prism_q2_0_subblock(
    thread const block_prism_q2_0& blk,
    uint sub_idx, // 0..3
    threadgroup half* sh_B,
    uint linear_tid)
{
    half d = blk.d;
    uint qs_offset = sub_idx * 8;
    thread const uint8_t* p = blk.qs + qs_offset;
    uint32_t q0 = read_u32_unaligned(p + 0);
    uint32_t q1 = read_u32_unaligned(p + 4);

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

// ============================================================================
// 4. UNIFIED 2D BLOCKMMA ENGINE CORE (64x64 Tile, 4 SIMDgroups, FP32 Precision)
// ============================================================================

template <bool DIRECT_HEAD_ROUTING, typename WeightBlockType, typename UnpackFunctor>
inline void block_mma_64x64_engine(
    device const half*          A,
    device const WeightBlockType* B,
    device half*                C,
    uint M,
    uint N,
    uint K,
    uint H,
    uint D,
    threadgroup half*           shmem,
    uint2 tg_id,
    uint simd_lane_id,
    uint simd_group_id,
    UnpackFunctor unpack_fn)
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Shared memory layout:
    // sh_A: [64][32] = 2048 halfs (4KB)
    // sh_B: [32][64] = 2048 halfs (4KB)
    threadgroup half* sh_A = shmem;
    threadgroup half* sh_B = shmem + 2048;

    // 4 SIMDgroups mapped in a 2x2 grid (32x32 sub-tile per SIMDgroup)
    uint sg_r = simd_group_id / 2; // 0 or 1
    uint sg_c = simd_group_id % 2; // 0 or 1
    uint sg_row_offset = sg_r * 32;
    uint sg_col_offset = sg_c * 32;

    // 16 accumulators: 4 rows x 4 cols of 8x8 matrix fragments with high-precision FP32 accumulation
    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        // 1. Cooperative coalesced loading of activation A (64 rows x 32 cols = 256 float4s)
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint idx = linear_tid * 2 + i; // 0..255
            uint r = idx / 4;              // 0..63
            uint c = (idx % 4) * 8;        // 0, 8, 16, 24
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[r * 32 + c]) = val;
        }

        // 2. Cooperative dequantization & SRAM staging of weight B (64 columns)
        if (linear_tid < 64) {
            uint b_col_idx = tg_col_start + linear_tid;
            if (b_col_idx < N) {
                unpack_fn(B, b_col_idx, kb, K, sh_B, linear_tid);
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B[k * 64 + linear_tid] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Hardware SIMDgroup Matrix Multiply-Accumulate (4 substeps of K=8)
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint k_off = ks * 8;
            simdgroup_matrix<half, 8, 8> a_frag[4];
            simdgroup_matrix<half, 8, 8> b_frag[4];

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                simdgroup_load(a_frag[r], &sh_A[(sg_row_offset + r * 8) * 32 + k_off], 32);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_load(b_frag[c], &sh_B[k_off * 64 + (sg_col_offset + c * 8)], 64);
            }

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_multiply_accumulate(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Staged Cooperative Writeback to global C via float shared memory (16KB)
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 128 threads cooperatively write 4096 elements to global C with full bounds protection
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint elem_idx = linear_tid * 32 + i;
        uint r = elem_idx / 64;
        uint c = elem_idx % 64;
        uint global_r = tg_row_start + r;
        uint global_c = tg_col_start + c;
        if (global_r < M && global_c < N) {
            if (DIRECT_HEAD_ROUTING) {
                // Direct-Head Routing: [M, N] -> [H, M, D] where N = H * D
                uint h = global_c / D;
                uint d_idx = global_c % D;
                C[(h * M + global_r) * D + d_idx] = (half)sh_Out[elem_idx];
            } else {
                // Standard GEMM Routing: [M, N]
                C[global_r * N + global_c] = (half)sh_Out[elem_idx];
            }
        }
    }
}

// ============================================================================
// 5. STANDARD GEMM ENTRY POINTS FOR ALL 4 QUANTIZATION FORMATS
// ============================================================================

// 5.1 QUANT_Q4_0 Standard GEMM
kernel void quant_router_gemm_q4_0_64x64(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_q4_0* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_q4_0 blk = b_ptr[col * nb + kb];
        unpack_q4_0_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.2 QUANT_MLX_4BIT Standard GEMM
kernel void quant_router_gemm_mlx_4bit_64x64(
    device const half*           A [[buffer(0)]],
    device const block_mlx_4bit* B [[buffer(1)]],
    device half*                 C [[buffer(2)]],
    constant uint&               M [[buffer(3)]],
    constant uint&               N [[buffer(4)]],
    constant uint&               K [[buffer(5)]],
    threadgroup half*            shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_mlx_4bit* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_mlx_4bit blk = b_ptr[col * nb + kb];
        unpack_mlx_4bit_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.3 QUANT_Q4_K Standard GEMM
kernel void quant_router_gemm_q4_k_64x64(
    device const half*         A [[buffer(0)]],
    device const block_q4_K*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_q4_K* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_q4_K blk = b_ptr[col * n_super + sb_idx];
        unpack_q4_K_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.4 QUANT_TERNARY_1_58 Standard GEMM
kernel void quant_router_gemm_ternary_1_58_64x64(
    device const half*               A [[buffer(0)]],
    device const block_ternary_1_58* B [[buffer(1)]],
    device half*                     C [[buffer(2)]],
    constant uint&                   M [[buffer(3)]],
    constant uint&                   N [[buffer(4)]],
    constant uint&                   K [[buffer(5)]],
    threadgroup half*                shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_ternary_1_58* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_ternary_1_58 blk = b_ptr[col * nb + kb];
        unpack_ternary_1_58_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.5 QUANT_VAR_RATE_AFFINE Standard GEMM
kernel void quant_router_gemm_var_rate_affine_64x64(
    device const half*                  A [[buffer(0)]],
    device const block_var_rate_affine* B [[buffer(1)]],
    device half*                        C [[buffer(2)]],
    constant uint&                      M [[buffer(3)]],
    constant uint&                      N [[buffer(4)]],
    constant uint&                      K [[buffer(5)]],
    threadgroup half*                   shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_var_rate_affine* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_var_rate_affine blk = b_ptr[col * n_super + sb_idx];
        unpack_var_rate_affine_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.6 QUANT_EXL3 Standard GEMM
kernel void quant_router_gemm_exl3_64x64(
    device const half*         A [[buffer(0)]],
    device const block_exl3*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_exl3* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_exl3 blk = b_ptr[col * n_super + sb_idx];
        unpack_exl3_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 5.7 QUANT_PRISM_Q2_0 Standard GEMM
kernel void quant_router_gemm_prism_q2_0_64x64(
    device const half*                    A [[buffer(0)]],
    device const block_prism_q2_0*        B [[buffer(1)]],
    device half*                          C [[buffer(2)]],
    constant uint&                        M [[buffer(3)]],
    constant uint&                        N [[buffer(4)]],
    constant uint&                        K [[buffer(5)]],
    threadgroup half*                     shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    auto unpack_fn = [](device const block_prism_q2_0* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_blocks = k_dim / 128;
        uint blk_idx = kb / 4;
        uint sub_idx = kb % 4;
        block_prism_q2_0 blk = b_ptr[col * n_blocks + blk_idx];
        unpack_prism_q2_0_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<false>(A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// ============================================================================
// 6. DIRECT-HEAD ROUTING GEMM ENTRY POINTS FOR ALL QUANTIZATION FORMATS
// ============================================================================

// 6.1 QUANT_Q4_0 Direct-Head GEMM
kernel void quant_router_head_gemm_q4_0_64x64(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             H [[buffer(4)]],
    constant uint&             D [[buffer(5)]],
    constant uint&             K [[buffer(6)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_q4_0* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_q4_0 blk = b_ptr[col * nb + kb];
        unpack_q4_0_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.2 QUANT_MLX_4BIT Direct-Head GEMM
kernel void quant_router_head_gemm_mlx_4bit_64x64(
    device const half*           A [[buffer(0)]],
    device const block_mlx_4bit* B [[buffer(1)]],
    device half*                 C [[buffer(2)]],
    constant uint&               M [[buffer(3)]],
    constant uint&               H [[buffer(4)]],
    constant uint&               D [[buffer(5)]],
    constant uint&               K [[buffer(6)]],
    threadgroup half*            shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_mlx_4bit* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_mlx_4bit blk = b_ptr[col * nb + kb];
        unpack_mlx_4bit_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.3 QUANT_Q4_K Direct-Head GEMM
kernel void quant_router_head_gemm_q4_k_64x64(
    device const half*         A [[buffer(0)]],
    device const block_q4_K*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             H [[buffer(4)]],
    constant uint&             D [[buffer(5)]],
    constant uint&             K [[buffer(6)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_q4_K* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_q4_K blk = b_ptr[col * n_super + sb_idx];
        unpack_q4_K_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.4 QUANT_TERNARY_1_58 Direct-Head GEMM
kernel void quant_router_head_gemm_ternary_1_58_64x64(
    device const half*               A [[buffer(0)]],
    device const block_ternary_1_58* B [[buffer(1)]],
    device half*                     C [[buffer(2)]],
    constant uint&                   M [[buffer(3)]],
    constant uint&                   H [[buffer(4)]],
    constant uint&                   D [[buffer(5)]],
    constant uint&                   K [[buffer(6)]],
    threadgroup half*                shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_ternary_1_58* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint nb = k_dim / 32;
        block_ternary_1_58 blk = b_ptr[col * nb + kb];
        unpack_ternary_1_58_block(blk, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.5 QUANT_VAR_RATE_AFFINE Direct-Head GEMM
kernel void quant_router_head_gemm_var_rate_affine_64x64(
    device const half*                  A [[buffer(0)]],
    device const block_var_rate_affine* B [[buffer(1)]],
    device half*                        C [[buffer(2)]],
    constant uint&                      M [[buffer(3)]],
    constant uint&                      H [[buffer(4)]],
    constant uint&                      D [[buffer(5)]],
    constant uint&                      K [[buffer(6)]],
    threadgroup half*                   shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_var_rate_affine* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_var_rate_affine blk = b_ptr[col * n_super + sb_idx];
        unpack_var_rate_affine_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.6 QUANT_EXL3 Direct-Head GEMM
kernel void quant_router_head_gemm_exl3_64x64(
    device const half*         A [[buffer(0)]],
    device const block_exl3*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             H [[buffer(4)]],
    constant uint&             D [[buffer(5)]],
    constant uint&             K [[buffer(6)]],
    threadgroup half*          shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_exl3* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_super = k_dim / 256;
        uint sb_idx = kb / 8;
        uint sub_idx = kb % 8;
        block_exl3 blk = b_ptr[col * n_super + sb_idx];
        unpack_exl3_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// 6.7 QUANT_PRISM_Q2_0 Direct-Head GEMM
kernel void quant_router_head_gemm_prism_q2_0_64x64(
    device const half*                    A [[buffer(0)]],
    device const block_prism_q2_0*        B [[buffer(1)]],
    device half*                          C [[buffer(2)]],
    constant uint&                        M [[buffer(3)]],
    constant uint&                        H [[buffer(4)]],
    constant uint&                        D [[buffer(5)]],
    constant uint&                        K [[buffer(6)]],
    threadgroup half*                     shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint N = H * D;
    auto unpack_fn = [](device const block_prism_q2_0* b_ptr, uint col, uint kb, uint k_dim, threadgroup half* sh_B, uint tid) {
        uint n_blocks = k_dim / 128;
        uint blk_idx = kb / 4;
        uint sub_idx = kb % 4;
        block_prism_q2_0 blk = b_ptr[col * n_blocks + blk_idx];
        unpack_prism_q2_0_subblock(blk, sub_idx, sh_B, tid);
    };

    block_mma_64x64_engine<true>(A, B, C, M, N, K, H, D, shmem, tg_id, simd_lane_id, simd_group_id, unpack_fn);
}

// ============================================================================
// 7. DEDICATED VECTOR ALU BRANCHLESS TERNARY 1.58-BIT GEMM ENGINES
// ============================================================================

// 7.1 True Branchless Vector ALU Ternary GEMM (Bypassing MMA multiplication)
kernel void quant_router_gemm_ternary_1_58_vec(
    device const half*               A [[buffer(0)]],
    device const block_ternary_1_58* B [[buffer(1)]],
    device half*                     C [[buffer(2)]],
    constant uint&                   M [[buffer(3)]],
    constant uint&                   N [[buffer(4)]],
    constant uint&                   K [[buffer(5)]],
    threadgroup half*                shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32][32] = (threadgroup half (*)[32][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_ternary_1_58 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_ternary_1_58 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            uint32_t q0 = q_curr.qs[0];
            uint32_t q1 = q_curr.qs[1];

            // Branchless sign extraction (0 -> -1.0h, 1 -> 0.0h, 2 -> +1.0h)
            half4 w_vec0[4], w_vec1[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t c0 = (q0 >> (j * 8 + 0)) & 3;
                uint8_t c1 = (q0 >> (j * 8 + 2)) & 3;
                uint8_t c2 = (q0 >> (j * 8 + 4)) & 3;
                uint8_t c3 = (q0 >> (j * 8 + 6)) & 3;
                w_vec0[j] = half4((c0 == 2 ? 1.0h : (c0 == 0 ? -1.0h : 0.0h)),
                                  (c1 == 2 ? 1.0h : (c1 == 0 ? -1.0h : 0.0h)),
                                  (c2 == 2 ? 1.0h : (c2 == 0 ? -1.0h : 0.0h)),
                                  (c3 == 2 ? 1.0h : (c3 == 0 ? -1.0h : 0.0h)));
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t c0 = (q1 >> (j * 8 + 0)) & 3;
                uint8_t c1 = (q1 >> (j * 8 + 2)) & 3;
                uint8_t c2 = (q1 >> (j * 8 + 4)) & 3;
                uint8_t c3 = (q1 >> (j * 8 + 6)) & 3;
                w_vec1[j] = half4((c0 == 2 ? 1.0h : (c0 == 0 ? -1.0h : 0.0h)),
                                  (c1 == 2 ? 1.0h : (c1 == 0 ? -1.0h : 0.0h)),
                                  (c2 == 2 ? 1.0h : (c2 == 0 ? -1.0h : 0.0h)),
                                  (c3 == 2 ? 1.0h : (c3 == 0 ? -1.0h : 0.0h)));
            }

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                if (tg_row_start + r < M) {
                    float row_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half4 a_vec = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][j * 4]);
                        row_sum += (float)dot(a_vec, w_vec0[j]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half4 a_vec = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16 + j * 4]);
                        row_sum += (float)dot(a_vec, w_vec1[j]);
                    }
                    acc[r] += row_sum * (float)d;
                }
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            q_curr = q_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 7.2 Direct-Head Branchless Vector ALU Ternary GEMM
kernel void quant_router_head_gemm_ternary_1_58_vec(
    device const half*               A [[buffer(0)]],
    device const block_ternary_1_58* B [[buffer(1)]],
    device half*                     C [[buffer(2)]],
    constant uint&                   M [[buffer(3)]],
    constant uint&                   H [[buffer(4)]],
    constant uint&                   D [[buffer(5)]],
    constant uint&                   K [[buffer(6)]],
    threadgroup half*                shmem [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint N = H * D;
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32][32] = (threadgroup half (*)[32][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_ternary_1_58 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_ternary_1_58 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            uint32_t q0 = q_curr.qs[0];
            uint32_t q1 = q_curr.qs[1];

            // Branchless sign extraction (0 -> -1.0h, 1 -> 0.0h, 2 -> +1.0h)
            half4 w_vec0[4], w_vec1[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t c0 = (q0 >> (j * 8 + 0)) & 3;
                uint8_t c1 = (q0 >> (j * 8 + 2)) & 3;
                uint8_t c2 = (q0 >> (j * 8 + 4)) & 3;
                uint8_t c3 = (q0 >> (j * 8 + 6)) & 3;
                w_vec0[j] = half4((c0 == 2 ? 1.0h : (c0 == 0 ? -1.0h : 0.0h)),
                                  (c1 == 2 ? 1.0h : (c1 == 0 ? -1.0h : 0.0h)),
                                  (c2 == 2 ? 1.0h : (c2 == 0 ? -1.0h : 0.0h)),
                                  (c3 == 2 ? 1.0h : (c3 == 0 ? -1.0h : 0.0h)));
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t c0 = (q1 >> (j * 8 + 0)) & 3;
                uint8_t c1 = (q1 >> (j * 8 + 2)) & 3;
                uint8_t c2 = (q1 >> (j * 8 + 4)) & 3;
                uint8_t c3 = (q1 >> (j * 8 + 6)) & 3;
                w_vec1[j] = half4((c0 == 2 ? 1.0h : (c0 == 0 ? -1.0h : 0.0h)),
                                  (c1 == 2 ? 1.0h : (c1 == 0 ? -1.0h : 0.0h)),
                                  (c2 == 2 ? 1.0h : (c2 == 0 ? -1.0h : 0.0h)),
                                  (c3 == 2 ? 1.0h : (c3 == 0 ? -1.0h : 0.0h)));
            }

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                if (tg_row_start + r < M) {
                    float row_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half4 a_vec = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][j * 4]);
                        row_sum += (float)dot(a_vec, w_vec0[j]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half4 a_vec = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16 + j * 4]);
                        row_sum += (float)dot(a_vec, w_vec1[j]);
                    }
                    acc[r] += row_sum * (float)d;
                }
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            q_curr = q_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        uint h = col_idx / D;
        uint d_idx = col_idx % D;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[(h * M + global_r) * D + d_idx] = (half)acc[r];
            }
        }
    }
}
