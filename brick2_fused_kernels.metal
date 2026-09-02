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
// 1. HARDWARE CEILING PROBES
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
// 2. KERNEL 1: gemm_mma_q4_0_64x64_baseline (Brick 1 Winner)
// Single-buffered, unpadded SRAM, standard linear Q4_0 memory layout
// ============================================================================
kernel void gemm_mma_q4_0_64x64_baseline(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // 16KB
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

    uint sg_r = simd_group_id / 2; // 0 or 1
    uint sg_c = simd_group_id % 2; // 0 or 1
    uint sg_row_offset = sg_r * 32;
    uint sg_col_offset = sg_c * 32;

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
        // 1. Cooperative coalesced loading of A: 64 rows x 32 cols = 256 float4s (128 threads load 2 float4s)
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
            *reinterpret_cast<threadgroup float4*>(&sh_A[r * 32 + c]) = val;
        }

        // 2. Cooperative coalesced loading & on-the-fly dequantization of B (64 columns)
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
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

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

// ============================================================================
// 3. KERNEL 2: gemm_mma_q4_0_64x64_double_buffered
// Double-buffered ping-pong staging in padded SRAM [2][64][36], linear Q4_0 B layout
// ============================================================================
kernel void gemm_mma_q4_0_64x64_double_buffered(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // 17408 bytes
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // sh_A: [2][64][36] = 4608 halfs = 9216 bytes (padded stride 36 prevents bank conflicts)
    // sh_B: [2][32][64] = 4096 halfs = 8192 bytes
    threadgroup half (*sh_A)[64][36] = (threadgroup half (*)[64][36])shmem;
    threadgroup half (*sh_B)[32][64] = (threadgroup half (*)[32][64])(shmem + 4608);

    uint sg_r = simd_group_id / 2;
    uint sg_c = simd_group_id % 2;
    uint sg_row_offset = sg_r * 32;
    uint sg_col_offset = sg_c * 32;

    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint num_k_blocks = K / 32;
    uint b_col_idx = tg_col_start + linear_tid;
    bool valid_b_col = (b_col_idx < N);

    auto load_A_tile = [&](uint buf_idx, uint kb) {
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
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c]) = val.xy;
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c + 4]) = val.zw;
        }
    };

    auto dequant_B_tile = [&](uint buf_idx, uint kb) {
        if (linear_tid < 64) {
            if (valid_b_col) {
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
                    sh_B[buf_idx][i * 4 + 0][linear_tid] = vl[i][0];
                    sh_B[buf_idx][i * 4 + 1][linear_tid] = vl[i][1];
                    sh_B[buf_idx][i * 4 + 2][linear_tid] = vl[i][2];
                    sh_B[buf_idx][i * 4 + 3][linear_tid] = vl[i][3];
                    sh_B[buf_idx][16 + i * 4 + 0][linear_tid] = vh[i][0];
                    sh_B[buf_idx][16 + i * 4 + 1][linear_tid] = vh[i][1];
                    sh_B[buf_idx][16 + i * 4 + 2][linear_tid] = vh[i][2];
                    sh_B[buf_idx][16 + i * 4 + 3][linear_tid] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B[buf_idx][k][linear_tid] = 0.0h;
                }
            }
        }
    };

    auto compute_mma = [&](uint buf_idx) {
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint k_off = ks * 8;
            simdgroup_matrix<half, 8, 8> a_frag[4];
            simdgroup_matrix<half, 8, 8> b_frag[4];

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                simdgroup_load(a_frag[r], &sh_A[buf_idx][sg_row_offset + r * 8][k_off], 36);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_load(b_frag[c], &sh_B[buf_idx][k_off][sg_col_offset + c * 8], 64);
            }

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_multiply_accumulate(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
                }
            }
        }
    };

    // Prologue: Load kb = 0
    load_A_tile(0, 0);
    dequant_B_tile(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Double-buffered loop
    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur = kb & 1;
        uint nxt = cur ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            load_A_tile(nxt, next_kb);
            dequant_B_tile(nxt, next_kb);
        }

        compute_mma(cur);

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Epilogue: Store results
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

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

// ============================================================================
// 4. KERNEL 3: gemm_mma_q4_0_64x64_fused_pipeline
// Fused 3-in-1 Pipeline:
// 1. 2D Block-Swizzled Q4_0 layout [N/64, K/32, 64, 18] in DRAM
// 2. 128-bit cooperative float4 burst loads into padded [64][36] SRAM
// 3. Asynchronous register load hoisting (q_prefetch) overlapping DRAM & MMA
// ============================================================================
kernel void gemm_mma_q4_0_64x64_fused_pipeline(
    device const half*         A          [[buffer(0)]],
    device const block_q4_0*   B_swizzled [[buffer(1)]], // 2D Swizzled blocks
    device half*               C          [[buffer(2)]],
    constant uint&             M          [[buffer(3)]],
    constant uint&             N          [[buffer(4)]],
    constant uint&             K          [[buffer(5)]],
    threadgroup half*          shmem      [[threadgroup(0)]], // 17408 bytes
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // sh_A: [2][64][36] = 4608 halfs = 9216 bytes (padded stride 36 prevents bank conflicts)
    // sh_B: [2][32][64] = 4096 halfs = 8192 bytes
    threadgroup half (*sh_A)[64][36] = (threadgroup half (*)[64][36])shmem;
    threadgroup half (*sh_B)[32][64] = (threadgroup half (*)[32][64])(shmem + 4608);

    uint sg_r = simd_group_id / 2;
    uint sg_c = simd_group_id % 2;
    uint sg_row_offset = sg_r * 32;
    uint sg_col_offset = sg_c * 32;

    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint num_k_blocks = K / 32;
    uint b_tile_n = tg_id.x;
    uint b_tile_base = b_tile_n * num_k_blocks * 64;
    bool valid_b_col = (tg_col_start + linear_tid < N);

    auto load_A_to_shmem = [&](uint buf_idx, uint kb) {
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
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c]) = val.xy;
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c + 4]) = val.zw;
        }
    };

    auto dequant_B_to_shmem = [&](uint buf_idx, block_q4_0 blk) {
        if (linear_tid < 64) {
            if (valid_b_col) {
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
                    sh_B[buf_idx][i * 4 + 0][linear_tid] = vl[i][0];
                    sh_B[buf_idx][i * 4 + 1][linear_tid] = vl[i][1];
                    sh_B[buf_idx][i * 4 + 2][linear_tid] = vl[i][2];
                    sh_B[buf_idx][i * 4 + 3][linear_tid] = vl[i][3];
                    sh_B[buf_idx][16 + i * 4 + 0][linear_tid] = vh[i][0];
                    sh_B[buf_idx][16 + i * 4 + 1][linear_tid] = vh[i][1];
                    sh_B[buf_idx][16 + i * 4 + 2][linear_tid] = vh[i][2];
                    sh_B[buf_idx][16 + i * 4 + 3][linear_tid] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B[buf_idx][k][linear_tid] = 0.0h;
                }
            }
        }
    };

    auto compute_mma = [&](uint buf_idx) {
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint k_off = ks * 8;
            simdgroup_matrix<half, 8, 8> a_frag[4];
            simdgroup_matrix<half, 8, 8> b_frag[4];

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                simdgroup_load(a_frag[r], &sh_A[buf_idx][sg_row_offset + r * 8][k_off], 36);
            }

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_load(b_frag[c], &sh_B[buf_idx][k_off][sg_col_offset + c * 8], 64);
            }

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_multiply_accumulate(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
                }
            }
        }
    };

    // --- PROLOGUE ---
    // 1. Stage kb = 0 directly to shmem buffer 0
    load_A_to_shmem(0, 0);
    block_q4_0 q_curr;
    if (linear_tid < 64 && valid_b_col) {
        q_curr = B_swizzled[b_tile_base + 0 * 64 + linear_tid];
    }
    dequant_B_to_shmem(0, q_curr);

    // 2. Prefetch kb = 1 into thread private registers
    float4 a_prefetch[2] = { float4(0.0f), float4(0.0f) };
    block_q4_0 q_prefetch;
    if (num_k_blocks > 1) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            uint idx = linear_tid * 2 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = 1 * 32 + c;
            if (global_r < M && global_c < K) {
                a_prefetch[i] = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
        }
        if (linear_tid < 64 && valid_b_col) {
            q_prefetch = B_swizzled[b_tile_base + 1 * 64 + linear_tid];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- MAIN LOOP ---
    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur = kb & 1;
        uint nxt = cur ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            // Stage prefetched data into shmem[nxt]
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                uint idx = linear_tid * 2 + i;
                uint r = idx / 4;
                uint c = (idx % 4) * 8;
                *reinterpret_cast<threadgroup float2*>(&sh_A[nxt][r][c]) = a_prefetch[i].xy;
                *reinterpret_cast<threadgroup float2*>(&sh_A[nxt][r][c + 4]) = a_prefetch[i].zw;
            }
            dequant_B_to_shmem(nxt, q_prefetch);
        }

        // Asynchronously issue next DRAM burst loads into registers while overlapped with MMA compute
        if (next_kb + 1 < num_k_blocks) {
            uint fetch_kb = next_kb + 1;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                uint idx = linear_tid * 2 + i;
                uint r = idx / 4;
                uint c = (idx % 4) * 8;
                uint global_r = tg_row_start + r;
                uint global_c = fetch_kb * 32 + c;
                a_prefetch[i] = float4(0.0f);
                if (global_r < M && global_c < K) {
                    a_prefetch[i] = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
                }
            }
            if (linear_tid < 64 && valid_b_col) {
                q_prefetch = B_swizzled[b_tile_base + fetch_kb * 64 + linear_tid];
            }
        }

        // Hardware MMA compute on cur buffer
        compute_mma(cur);

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // --- EPILOGUE: STORE ---
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

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
