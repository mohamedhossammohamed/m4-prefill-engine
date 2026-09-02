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
// HARDWARE CEILING PROBES
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
// OPTION A: ENHANCED VECTOR ALU KERNELS
// ============================================================================

// --- 1. Baseline: pipe_gemm_q4_0_32x32 (Vector ALU fma(half4) with 32-row scalar loop) ---
kernel void pipe_gemm_q4_0_32x32(
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

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s = (p0 + p1) + (p2 + p3);
                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
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

// --- 2. Enhanced: pipe_gemm_q4_0_64x32 (Vector ALU 64 rows x 32 cols) ---
kernel void pipe_gemm_q4_0_64x32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][64][32] = 8KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;

    float acc[64];
    #pragma unroll
    for (int r = 0; r < 64; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint idx = simd_lane_id * 8 + i;
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
            for (int r = 0; r < 64; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s = (p0 + p1) + (p2 + p3);
                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
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
        for (int r = 0; r < 64; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// --- 3. Enhanced: pipe_gemm_q4_0_128x32 (Vector ALU 128 rows x 32 cols) ---
kernel void pipe_gemm_q4_0_128x32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][128][32] = 16KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 128;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[128][32] = (threadgroup half (*)[128][32])shmem;

    float acc[128];
    #pragma unroll
    for (int r = 0; r < 128; r++) {
        acc[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint idx = simd_lane_id * 16 + i;
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
            for (int r = 0; r < 128; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s = (p0 + p1) + (p2 + p3);
                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
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
        for (int r = 0; r < 128; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// ============================================================================
// OPTION B: HARDWARE MATRIX UNITS VIA simdgroup_matrix (Apple AMX/MMA)
// ============================================================================

// --- 4. gemm_mma_q4_0_32x32 (1 SIMDgroup = 32 threads, 32x32 tile via sixteen 8x8 fragments, FP32 acc) ---
kernel void gemm_mma_q4_0_32x32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // 4KB (holds sh_A[32][32] + sh_B[32][32], then sh_Out[32][32] float)
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    threadgroup half* sh_A = shmem;             // [32][32] = 1024 halfs (2KB)
    threadgroup half* sh_B = shmem + 1024;      // [32][32] = 1024 halfs (2KB)

    // 16 accumulators: 4 rows x 4 cols of 8x8 matrix fragments with high-precision FP32 accumulation
    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        // 1. Cooperative load of tile A (32x32 = 1024 halfs = 256 float4s)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint idx = simd_lane_id * 8 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[r * 32 + c]) = val;
        }

        // 2. On-the-fly Q4_0 dequantization of tile B (32 columns x 32 rows in K)
        if (valid_col) {
            block_q4_0 blk = B[col_idx * num_k_blocks + kb];
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
                sh_B[(i * 4 + 0) * 32 + simd_lane_id] = vl[i][0];
                sh_B[(i * 4 + 1) * 32 + simd_lane_id] = vl[i][1];
                sh_B[(i * 4 + 2) * 32 + simd_lane_id] = vl[i][2];
                sh_B[(i * 4 + 3) * 32 + simd_lane_id] = vl[i][3];
                sh_B[(16 + i * 4 + 0) * 32 + simd_lane_id] = vh[i][0];
                sh_B[(16 + i * 4 + 1) * 32 + simd_lane_id] = vh[i][1];
                sh_B[(16 + i * 4 + 2) * 32 + simd_lane_id] = vh[i][2];
                sh_B[(16 + i * 4 + 3) * 32 + simd_lane_id] = vh[i][3];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                sh_B[k * 32 + simd_lane_id] = 0.0h;
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
                simdgroup_load(a_frag[r], &sh_A[r * 8 * 32 + k_off], 32);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_load(b_frag[c], &sh_B[k_off * 32 + c * 8], 32);
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

    // 4. Store accumulated FP32 results to shared memory then copy to C with full bounds protection
    threadgroup float* sh_Out = (threadgroup float*)shmem; // [32][32] floats = 4KB
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[r * 8 * 32 + c * 8], 32);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (int r = 0; r < 32; r++) {
        uint global_r = tg_row_start + r;
        if (global_r < M && valid_col) {
            C[global_r * N + col_idx] = (half)sh_Out[r * 32 + simd_lane_id];
        }
    }
}

// --- 5. gemm_mma_q4_0_64x64 (4 SIMDgroups = 128 threads, cooperative coalesced loading + 64x64 tile, FP32 acc) ---
kernel void gemm_mma_q4_0_64x64(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // 16KB (during compute: sh_A 4KB + sh_B 4KB; during store: sh_Out 16KB)
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // sh_A: [64][32] = 2048 halfs (4KB)
    // sh_B: [32][64] = 2048 halfs (4KB)
    threadgroup half* sh_A = shmem;
    threadgroup half* sh_B = shmem + 2048;

    // 4 SIMDgroups mapped in a 2x2 grid:
    // SIMD 0: row 0..31, col 0..31
    // SIMD 1: row 0..31, col 32..63
    // SIMD 2: row 32..63, col 0..31
    // SIMD 3: row 32..63, col 32..63
    uint sg_r = simd_group_id / 2; // 0 or 1
    uint sg_c = simd_group_id % 2; // 0 or 1
    uint sg_row_offset = sg_r * 32;
    uint sg_col_offset = sg_c * 32;

    // Each SIMDgroup maintains sixteen 8x8 accumulators covering its 32x32 sub-tile with FP32 precision
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
        // 1. Cooperative coalesced loading of A: 64 rows x 32 cols = 2048 halfs = 256 float4s
        // 128 threads: each thread loads 2 float4s (16 halfs)
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

        // 2. Cooperative coalesced loading & on-the-fly dequantization of B (64 columns)
        // Threads 0..63 unpack columns 0..63
        if (linear_tid < 64) {
            uint b_col_idx = tg_col_start + linear_tid;
            if (b_col_idx < N) {
                block_q4_0 blk = B[b_col_idx * num_k_blocks + kb];
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
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B[k * 64 + linear_tid] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. SIMDgroup Matrix Multiply-Accumulate (4 substeps of K=8)
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

    // 4. Staged Cooperative Writeback to global C via float shared memory
    // shmem has 16KB = 4096 floats = exactly [64][64] elements!
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 128 threads cooperatively write 4096 elements to global C with full boundary guards
    // Each thread writes 4096 / 128 = 32 elements
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint elem_idx = linear_tid * 32 + i;
        uint r = elem_idx / 64;
        uint c = elem_idx % 64;
        uint global_r = tg_row_start + r;
        uint global_c = tg_col_start + c;
        if (global_r < M && global_c < N) {
            C[global_r * N + global_c] = (half)sh_Out[elem_idx];
        }
    }
}
