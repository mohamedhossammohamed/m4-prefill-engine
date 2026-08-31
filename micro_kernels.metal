#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// ----------------------------------------------------------------------------
// 1. HARDWARE PROBES
// ----------------------------------------------------------------------------
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
    half4 a0 = half4(1.001h, 1.002h, 1.003h, 1.004h);
    half4 a1 = half4(1.005h, 1.006h, 1.007h, 1.008h);
    half4 b0 = half4(0.999h, 0.998h, 0.997h, 0.996h);
    half4 b1 = half4(0.995h, 0.994h, 0.993h, 0.992h);
    half4 c0 = half4(0.001h);
    half4 c1 = half4(0.002h);

    #pragma unroll(128)
    for (int i = 0; i < 128; i++) {
        c0 = fma(a0, b0, c0);
        c1 = fma(a1, b1, c1);
        a0 = fma(b0, c0, a0);
        a1 = fma(b1, c1, a1);
        b0 = fma(c0, a0, b0);
        b1 = fma(c1, a1, b1);
    }

    if (id == 0) {
        out[0] = c0[0] + c1[0] + a0[0] + a1[0] + b0[0] + b1[0];
    }
}

// ----------------------------------------------------------------------------
// 2. PRODUCTION BASELINE: LLAMA.CPP STYLE mul_mm (Threadgroup Drawer Staging)
// ----------------------------------------------------------------------------
kernel void llamacpp_style_mul_mm_q4_0(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [64 x 32] = 4KB
    threadgroup half*          sh_B [[threadgroup(1)]], // [32 x 32] = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (simd_lane_id / 8) * 8 + (simd_group_id * 32);
    uint thread_c = (simd_lane_id % 8) * 4;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            uint idx = linear_tid * 32 + i;
            uint r = idx / 32;
            uint c = idx % 32;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            if (global_r < M && global_c < K) {
                sh_A[r * 32 + c] = A[global_r * K + global_c];
            } else {
                sh_A[r * 32 + c] = 0.0h;
            }
        }

        if (linear_tid < 32) {
            uint b_col_idx = tg_col_start + linear_tid;
            if (b_col_idx < N) {
                block_q4_0 blk = B[b_col_idx * num_k_blocks + kb];
                half d = blk.d;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = blk.qs[i];
                    int v0 = (int)(byte_val & 0x0F) - 8;
                    int v1 = (int)(byte_val >> 4) - 8;
                    sh_B[i * 32 + linear_tid] = (half)((float)v0 * (float)d);
                    sh_B[(i + 16) * 32 + linear_tid] = (half)((float)v1 * (float)d);
                }
            } else {
                for (int i = 0; i < 32; i++) {
                    sh_B[i * 32 + linear_tid] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            half a_reg[8];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a_reg[r] = sh_A[(thread_r + r) * 32 + k];
            }

            half b_reg[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                b_reg[c] = sh_B[k * 32 + (thread_c + c)];
            }

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    acc[r][c] += (float)a_reg[r] * (float)b_reg[c];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 3. FUSED COLUMN BROADCAST: TILE 16x1
// ----------------------------------------------------------------------------
kernel void fused_col_16x1(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [16 x 32] = 1KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 16;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + linear_tid;
    bool valid_col = (col_idx < N);

    float acc[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        acc[r] = 0.0f;
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid / 4;
        uint c_offset = (linear_tid % 4) * 8;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint global_c = kb * 32 + c_offset + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + c_offset + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            block_q4_0 blk = B[col_idx * num_k_blocks + kb];
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v[8];
            v[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 16; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 28]);

                acc[r] += (float)(
                    dot(a0, v[0]) + dot(a1, v[1]) + dot(a2, v[2]) + dot(a3, v[3]) +
                    dot(a4, v[4]) + dot(a5, v[5]) + dot(a6, v[6]) + dot(a7, v[7])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 4. FUSED COLUMN BROADCAST: TILE 16x2
// ----------------------------------------------------------------------------
kernel void fused_col_16x2(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [16 x 32] = 1KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 16;
    uint tg_col_start = tg_id.x * 128;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col0 = tg_col_start + linear_tid * 2;
    uint col1 = col0 + 1;
    bool valid_col0 = (col0 < N);
    bool valid_col1 = (col1 < N);

    float acc0[16];
    float acc1[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid / 4;
        uint c_offset = (linear_tid % 4) * 8;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint global_c = kb * 32 + c_offset + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + c_offset + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        half4 v0[8];
        half4 v1[8];

        if (valid_col0) {
            block_q4_0 blk0 = B[col0 * num_k_blocks + kb];
            half d0 = blk0.d;
            half4 hd0 = half4(d0);
            half4 h_off0 = half4(-8.0h * d0);

            uint w0 = read_u32_unaligned(blk0.qs + 0);
            uint w1 = read_u32_unaligned(blk0.qs + 4);
            uint w2 = read_u32_unaligned(blk0.qs + 8);
            uint w3 = read_u32_unaligned(blk0.qs + 12);

            v0[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
        }

        if (valid_col1) {
            block_q4_0 blk1 = B[col1 * num_k_blocks + kb];
            half d1 = blk1.d;
            half4 hd1 = half4(d1);
            half4 h_off1 = half4(-8.0h * d1);

            uint w0 = read_u32_unaligned(blk1.qs + 0);
            uint w1 = read_u32_unaligned(blk1.qs + 4);
            uint w2 = read_u32_unaligned(blk1.qs + 8);
            uint w3 = read_u32_unaligned(blk1.qs + 12);

            v1[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
        }

        #pragma unroll
        for (int r = 0; r < 16; r++) {
            half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 0]);
            half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 4]);
            half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 8]);
            half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 12]);
            half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 16]);
            half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 20]);
            half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 24]);
            half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 28]);

            if (valid_col0) {
                acc0[r] += (float)(
                    dot(a0, v0[0]) + dot(a1, v0[1]) + dot(a2, v0[2]) + dot(a3, v0[3]) +
                    dot(a4, v0[4]) + dot(a5, v0[5]) + dot(a6, v0[6]) + dot(a7, v0[7])
                );
            }
            if (valid_col1) {
                acc1[r] += (float)(
                    dot(a0, v1[0]) + dot(a1, v1[1]) + dot(a2, v1[2]) + dot(a3, v1[3]) +
                    dot(a4, v1[4]) + dot(a5, v1[5]) + dot(a6, v1[6]) + dot(a7, v1[7])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_r = tg_row_start + r;
        if (global_r < M) {
            if (valid_col0) C[global_r * N + col0] = (half)acc0[r];
            if (valid_col1) C[global_r * N + col1] = (half)acc1[r];
        }
    }
}

// ----------------------------------------------------------------------------
// 5. FUSED COLUMN BROADCAST: TILE 32x1
// ----------------------------------------------------------------------------
kernel void fused_col_32x1(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [32 x 32] = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + linear_tid;
    bool valid_col = (col_idx < N);

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid / 2;
        uint c_offset = (linear_tid % 2) * 16;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint global_c = kb * 32 + c_offset + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + c_offset + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            block_q4_0 blk = B[col_idx * num_k_blocks + kb];
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v[8];
            v[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 28]);

                acc[r] += (float)(
                    dot(a0, v[0]) + dot(a1, v[1]) + dot(a2, v[2]) + dot(a3, v[3]) +
                    dot(a4, v[4]) + dot(a5, v[5]) + dot(a6, v[6]) + dot(a7, v[7])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
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

// ----------------------------------------------------------------------------
// 6. FUSED COLUMN BROADCAST: TILE 32x2
// ----------------------------------------------------------------------------
kernel void fused_col_32x2(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [32 x 32] = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 128;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col0 = tg_col_start + linear_tid * 2;
    uint col1 = col0 + 1;
    bool valid_col0 = (col0 < N);
    bool valid_col1 = (col1 < N);

    float acc0[32];
    float acc1[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid / 2;
        uint c_offset = (linear_tid % 2) * 16;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint global_c = kb * 32 + c_offset + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + c_offset + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        half4 v0[8];
        half4 v1[8];

        if (valid_col0) {
            block_q4_0 blk0 = B[col0 * num_k_blocks + kb];
            half d0 = blk0.d;
            half4 hd0 = half4(d0);
            half4 h_off0 = half4(-8.0h * d0);

            uint w0 = read_u32_unaligned(blk0.qs + 0);
            uint w1 = read_u32_unaligned(blk0.qs + 4);
            uint w2 = read_u32_unaligned(blk0.qs + 8);
            uint w3 = read_u32_unaligned(blk0.qs + 12);

            v0[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
            v0[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd0, h_off0);
        }

        if (valid_col1) {
            block_q4_0 blk1 = B[col1 * num_k_blocks + kb];
            half d1 = blk1.d;
            half4 hd1 = half4(d1);
            half4 h_off1 = half4(-8.0h * d1);

            uint w0 = read_u32_unaligned(blk1.qs + 0);
            uint w1 = read_u32_unaligned(blk1.qs + 4);
            uint w2 = read_u32_unaligned(blk1.qs + 8);
            uint w3 = read_u32_unaligned(blk1.qs + 12);

            v1[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
            v1[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd1, h_off1);
        }

        #pragma unroll
        for (int r = 0; r < 32; r++) {
            half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 0]);
            half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 4]);
            half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 8]);
            half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 12]);
            half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 16]);
            half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 20]);
            half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 24]);
            half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 28]);

            if (valid_col0) {
                acc0[r] += (float)(
                    dot(a0, v0[0]) + dot(a1, v0[1]) + dot(a2, v0[2]) + dot(a3, v0[3]) +
                    dot(a4, v0[4]) + dot(a5, v0[5]) + dot(a6, v0[6]) + dot(a7, v0[7])
                );
            }
            if (valid_col1) {
                acc1[r] += (float)(
                    dot(a0, v1[0]) + dot(a1, v1[1]) + dot(a2, v1[2]) + dot(a3, v1[3]) +
                    dot(a4, v1[4]) + dot(a5, v1[5]) + dot(a6, v1[6]) + dot(a7, v1[7])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 32; r++) {
        uint global_r = tg_row_start + r;
        if (global_r < M) {
            if (valid_col0) C[global_r * N + col0] = (half)acc0[r];
            if (valid_col1) C[global_r * N + col1] = (half)acc1[r];
        }
    }
}

// ----------------------------------------------------------------------------
// 7. FUSED COLUMN BROADCAST: TILE 64x1
// ----------------------------------------------------------------------------
kernel void fused_col_64x1(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [64 x 32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + linear_tid;
    bool valid_col = (col_idx < N);

    float acc[64];
    #pragma unroll
    for (int r = 0; r < 64; r++) {
        acc[r] = 0.0f;
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint global_c = kb * 32 + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            block_q4_0 blk = B[col_idx * num_k_blocks + kb];
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v[8];
            v[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v[4] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[5] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[6] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v[7] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 64; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r * 32 + 28]);

                acc[r] += (float)(
                    dot(a0, v[0]) + dot(a1, v[1]) + dot(a2, v[2]) + dot(a3, v[3]) +
                    dot(a4, v[4]) + dot(a5, v[5]) + dot(a6, v[6]) + dot(a7, v[7])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 64; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 8. 2D REGISTER TILE VARIANTS (4x4, 8x4, 8x8, 4x8, 16x4)
// ----------------------------------------------------------------------------
kernel void fused_q4_gemm_4x4(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (linear_tid / 8) * 4;
    uint thread_c = (linear_tid % 8) * 4;

    float acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        if (linear_tid < 32) {
            uint r_load = linear_tid;
            uint global_r = tg_row_start + r_load;
            bool valid_row = (global_r < M);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint global_c = kb * 32 + i * 4;
                half4 a_val = half4(0.0h);
                if (valid_row && global_c < K) {
                    a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
                }
                *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        block_q4_0 blk[4];
        uint w[4][4];
        half4 hd[4];
        half4 h_off[4];

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                blk[c] = B[b_col * num_k_blocks + kb];
                half d = blk[c].d;
                hd[c] = half4(d);
                h_off[c] = half4(-8.0h * d);

                w[c][0] = read_u32_unaligned(blk[c].qs + 0);
                w[c][1] = read_u32_unaligned(blk[c].qs + 4);
                w[c][2] = read_u32_unaligned(blk[c].qs + 8);
                w[c][3] = read_u32_unaligned(blk[c].qs + 12);
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[4];
            half4 a_high[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + 16 + step * 4]);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                uint cur_w = w[c][step];
                half4 v_low  = fma(half4(as_type<uchar4>(cur_w & 0x0F0F0F0Fu)), hd[c], h_off[c]);
                half4 v_high = fma(half4(as_type<uchar4>((cur_w >> 4) & 0x0F0F0F0Fu)), hd[c], h_off[c]);

                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    acc[r][c] += (float)(dot(a_low[r], v_low) + dot(a_high[r], v_high));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}

kernel void fused_q4_gemm_8x4(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (simd_lane_id / 8) * 8 + (simd_group_id * 32);
    uint thread_c = (simd_lane_id % 8) * 4;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint global_c = kb * 32 + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        block_q4_0 blk[4];
        uint w[4][4];
        half4 hd[4];
        half4 h_off[4];

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                blk[c] = B[b_col * num_k_blocks + kb];
                half d = blk[c].d;
                hd[c] = half4(d);
                h_off[c] = half4(-8.0h * d);

                w[c][0] = read_u32_unaligned(blk[c].qs + 0);
                w[c][1] = read_u32_unaligned(blk[c].qs + 4);
                w[c][2] = read_u32_unaligned(blk[c].qs + 8);
                w[c][3] = read_u32_unaligned(blk[c].qs + 12);
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[8];
            half4 a_high[8];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + 16 + step * 4]);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                uint cur_w = w[c][step];
                half4 v_low  = fma(half4(as_type<uchar4>(cur_w & 0x0F0F0F0Fu)), hd[c], h_off[c]);
                half4 v_high = fma(half4(as_type<uchar4>((cur_w >> 4) & 0x0F0F0F0Fu)), hd[c], h_off[c]);

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    acc[r][c] += (float)(dot(a_low[r], v_low) + dot(a_high[r], v_high));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}

kernel void fused_q4_gemm_8x8(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (linear_tid / 8) * 8;
    uint thread_c = (linear_tid % 8) * 8;

    float acc[8][8];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint r_load = linear_tid;
        uint global_r = tg_row_start + r_load;
        bool valid_row = (global_r < M);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint global_c = kb * 32 + i * 4;
            half4 a_val = half4(0.0h);
            if (valid_row && global_c < K) {
                a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        block_q4_0 blk[8];
        uint w[8][4];
        half4 hd[8];
        half4 h_off[8];

        #pragma unroll
        for (int c = 0; c < 8; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                blk[c] = B[b_col * num_k_blocks + kb];
                half d = blk[c].d;
                hd[c] = half4(d);
                h_off[c] = half4(-8.0h * d);

                w[c][0] = read_u32_unaligned(blk[c].qs + 0);
                w[c][1] = read_u32_unaligned(blk[c].qs + 4);
                w[c][2] = read_u32_unaligned(blk[c].qs + 8);
                w[c][3] = read_u32_unaligned(blk[c].qs + 12);
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[8];
            half4 a_high[8];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + 16 + step * 4]);
            }

            #pragma unroll
            for (int c = 0; c < 8; c++) {
                uint cur_w = w[c][step];
                half4 v_low  = fma(half4(as_type<uchar4>(cur_w & 0x0F0F0F0Fu)), hd[c], h_off[c]);
                half4 v_high = fma(half4(as_type<uchar4>((cur_w >> 4) & 0x0F0F0F0Fu)), hd[c], h_off[c]);

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    acc[r][c] += (float)(dot(a_low[r], v_low) + dot(a_high[r], v_high));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}

kernel void fused_q4_gemm_4x8(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (linear_tid / 8) * 4;
    uint thread_c = (linear_tid % 8) * 8;

    float acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        if (linear_tid < 32) {
            uint r_load = linear_tid;
            uint global_r = tg_row_start + r_load;
            bool valid_row = (global_r < M);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint global_c = kb * 32 + i * 4;
                half4 a_val = half4(0.0h);
                if (valid_row && global_c < K) {
                    a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
                }
                *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        block_q4_0 blk[8];
        uint w[8][4];
        half4 hd[8];
        half4 h_off[8];

        #pragma unroll
        for (int c = 0; c < 8; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                blk[c] = B[b_col * num_k_blocks + kb];
                half d = blk[c].d;
                hd[c] = half4(d);
                h_off[c] = half4(-8.0h * d);

                w[c][0] = read_u32_unaligned(blk[c].qs + 0);
                w[c][1] = read_u32_unaligned(blk[c].qs + 4);
                w[c][2] = read_u32_unaligned(blk[c].qs + 8);
                w[c][3] = read_u32_unaligned(blk[c].qs + 12);
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[4];
            half4 a_high[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + 16 + step * 4]);
            }

            #pragma unroll
            for (int c = 0; c < 8; c++) {
                uint cur_w = w[c][step];
                half4 v_low  = fma(half4(as_type<uchar4>(cur_w & 0x0F0F0F0Fu)), hd[c], h_off[c]);
                half4 v_high = fma(half4(as_type<uchar4>((cur_w >> 4) & 0x0F0F0F0Fu)), hd[c], h_off[c]);

                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    acc[r][c] += (float)(dot(a_low[r], v_low) + dot(a_high[r], v_high));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}

kernel void fused_q4_gemm_16x4(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 128;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (linear_tid / 8) * 16;
    uint thread_c = (linear_tid % 8) * 4;

    float acc[16][4];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = 0.0f;
        }
    }

    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int row_pass = 0; row_pass < 2; row_pass++) {
            uint r_load = linear_tid + row_pass * 64;
            uint global_r = tg_row_start + r_load;
            bool valid_row = (global_r < M);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint global_c = kb * 32 + i * 4;
                half4 a_val = half4(0.0h);
                if (valid_row && global_c < K) {
                    a_val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
                }
                *reinterpret_cast<threadgroup half4*>(&sh_A[r_load * 32 + i * 4]) = a_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        block_q4_0 blk[4];
        uint w[4][4];
        half4 hd[4];
        half4 h_off[4];

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                blk[c] = B[b_col * num_k_blocks + kb];
                half d = blk[c].d;
                hd[c] = half4(d);
                h_off[c] = half4(-8.0h * d);

                w[c][0] = read_u32_unaligned(blk[c].qs + 0);
                w[c][1] = read_u32_unaligned(blk[c].qs + 4);
                w[c][2] = read_u32_unaligned(blk[c].qs + 8);
                w[c][3] = read_u32_unaligned(blk[c].qs + 12);
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[16];
            half4 a_high[16];
            #pragma unroll
            for (int r = 0; r < 16; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[(thread_r + r) * 32 + 16 + step * 4]);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                uint cur_w = w[c][step];
                half4 v_low  = fma(half4(as_type<uchar4>(cur_w & 0x0F0F0F0Fu)), hd[c], h_off[c]);
                half4 v_high = fma(half4(as_type<uchar4>((cur_w >> 4) & 0x0F0F0F0Fu)), hd[c], h_off[c]);

                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    acc[r][c] += (float)(dot(a_low[r], v_low) + dot(a_high[r], v_high));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_r = tg_row_start + thread_r + r;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint global_c = tg_col_start + thread_c + c;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = (half)acc[r][c];
            }
        }
    }
}
