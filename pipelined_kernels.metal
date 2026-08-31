#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA TYPES & HELPERS
// ============================================================================
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// ============================================================================
// 1. HARDWARE PROBES
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

// ============================================================================
// 2. NAIVE GEMM REFERENCE BASELINE
// ============================================================================
kernel void naive_q4_0_gemm(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;

    if (row >= M || col >= N) return;

    uint nb = K / 32;
    device const block_q4_0* b_col = B + col * nb;
    device const half* a_row = A + row * K;

    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        block_q4_0 blk = b_col[b];
        float d = (float)blk.d;
        uint a_offset = b * 32;

        for (int i = 0; i < 16; i++) {
            uint8_t byte_val = blk.qs[i];
            int v0 = (int)(byte_val & 0x0F) - 8;
            int v1 = (int)(byte_val >> 4) - 8;
            acc += (float)a_row[a_offset + i] * ((float)v0 * d);
            acc += (float)a_row[a_offset + i + 16] * ((float)v1 * d);
        }
    }
    C[row * N + col] = (half)acc;
}

// ============================================================================
// 3. PRODUCTION BASELINE: LLAMA.CPP STYLE mul_mm (Drawer Staging)
// ============================================================================
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

// ============================================================================
// 4. 1D COLUMN-BROADCAST PIPELINED KERNELS
// ============================================================================

// 32x32 Double Buffered (1 SIMDgroup = 32 threads)
kernel void pipe_gemm_32x32_double(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][32][32] = 4KB
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
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
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

// 32x32 Single Buffered
kernel void pipe_gemm_32x32_single(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [32][32] = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32] = (threadgroup half (*)[32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    for (uint kb = 0; kb < num_k_blocks; kb++) {
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
            *reinterpret_cast<threadgroup float4*>(&sh_A[r][c]) = val;
        }

        block_q4_0 blk;
        if (valid_col) {
            blk = B[col_idx * num_k_blocks + kb];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
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

// 64x32 Double Buffered (2 SIMDgroups = 64 threads)
kernel void pipe_gemm_64x32_double(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = linear_tid * 4 + i;
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
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 64x32 Single Buffered
kernel void pipe_gemm_64x32_single(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [64][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[32] = (threadgroup half (*)[32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = linear_tid * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[r][c]) = val;
        }

        block_q4_0 blk;
        if (valid_col) {
            blk = B[col_idx * num_k_blocks + kb];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 64x32 Tile with 64-bit Vector Width (half4)
kernel void pipe_gemm_64x32_vec64(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint idx = linear_tid * 8 + i;
            uint r = idx / 8;
            uint c = (idx % 8) * 4;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            half4 val = half4(0.0h);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const half4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 64x32 Tile with 32-bit Vector Width (half2)
kernel void pipe_gemm_64x32_vec32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint idx = linear_tid * 16 + i;
            uint r = idx / 16;
            uint c = (idx % 16) * 2;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            half2 val = half2(0.0h);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const half2*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup half2*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 64x64 Double Buffered (4 SIMDgroups = 128 threads)
kernel void pipe_gemm_64x64_double(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_sub = (simd_group_id >= 2) ? 32 : 0;
    uint col_idx = tg_col_start + col_sub + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = (simd_group_id & 1) * 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint idx = linear_tid * 2 + i;
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
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 64x64 Single Buffered
kernel void pipe_gemm_64x64_single(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [64][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_sub = (simd_group_id >= 2) ? 32 : 0;
    uint col_idx = tg_col_start + col_sub + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = (simd_group_id & 1) * 32;

    threadgroup half (*sh_A)[32] = (threadgroup half (*)[32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint idx = linear_tid * 2 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[r][c]) = val;
        }

        block_q4_0 blk;
        if (valid_col) {
            blk = B[col_idx * num_k_blocks + kb];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 128x32 Double Buffered (4 SIMDgroups = 128 threads)
kernel void pipe_gemm_128x32_double(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][128][32] = 16KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 128;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[128][32] = (threadgroup half (*)[128][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = linear_tid * 4 + i;
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
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur_buf = nxt_buf;
        q_curr = q_next;
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// 128x32 Single Buffered
kernel void pipe_gemm_128x32_single(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [128][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 128;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;
    uint local_r_offset = simd_group_id * 32;

    threadgroup half (*sh_A)[32] = (threadgroup half (*)[32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = linear_tid * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[r][c]) = val;
        }

        block_q4_0 blk;
        if (valid_col) {
            blk = B[col_idx * num_k_blocks + kb];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_col) {
            half d = blk.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(blk.qs + 0);
            uint w1 = read_u32_unaligned(blk.qs + 4);
            uint w2 = read_u32_unaligned(blk.qs + 8);
            uint w3 = read_u32_unaligned(blk.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                uint r_sh = local_r_offset + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[r_sh][28]);

                acc[r] += (float)(
                    dot(a0, v_low[0])  + dot(a1, v_low[1])  + dot(a2, v_low[2])  + dot(a3, v_low[3]) +
                    dot(a4, v_high[0]) + dot(a5, v_high[1]) + dot(a6, v_high[2]) + dot(a7, v_high[3])
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + local_r_offset + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// ============================================================================
// 5. 2D REGISTER-TILED DOUBLE-BUFFERED PIPELINED KERNELS
// ============================================================================

// 2D Tile: 64x32 with 8x4 register tiling per thread (64 threads = 2 SIMDgroups)
kernel void pipe_gemm_64x32_2d_double(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint thread_r = (simd_lane_id / 8) * 8 + (simd_group_id * 32);
    uint thread_c = (simd_lane_id % 8) * 4;

    uint num_k_blocks = K / 32;
    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = 0.0f;
        }
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = linear_tid * 4 + i;
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

    auto load_B = [&](block_q4_0 q[4], uint kb) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint b_col = tg_col_start + thread_c + c;
            if (b_col < N) {
                q[c] = B[b_col * num_k_blocks + kb];
            }
        }
    };

    block_q4_0 q_curr[4];
    block_q4_0 q_next[4];

    load_A(0, 0);
    load_B(q_curr, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur_buf = kb & 1;
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            load_B(q_next, next_kb);
        }

        uint w[4][4];
        half4 hd[4];
        half4 h_off[4];

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            half d = q_curr[c].d;
            hd[c] = half4(d);
            h_off[c] = half4(-8.0h * d);

            w[c][0] = read_u32_unaligned(q_curr[c].qs + 0);
            w[c][1] = read_u32_unaligned(q_curr[c].qs + 4);
            w[c][2] = read_u32_unaligned(q_curr[c].qs + 8);
            w[c][3] = read_u32_unaligned(q_curr[c].qs + 12);
        }

        #pragma unroll
        for (int step = 0; step < 4; step++) {
            half4 a_low[8];
            half4 a_high[8];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a_low[r]  = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][thread_r + r][step * 4]);
                a_high[r] = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][thread_r + r][16 + step * 4]);
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
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            q_curr[c] = q_next[c];
        }
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
