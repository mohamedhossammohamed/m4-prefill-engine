#include <metal_stdlib>
using namespace metal;

// Standard Q4_0 block: 32 weights in 16 bytes (two 4-bit nibbles per byte) + 1 FP16 scale
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

// ============================================================================
// 1. HARDWARE PROBE: Raw Memory Bandwidth (Unified Memory Streaming)
// ============================================================================
kernel void probe_memory_bandwidth(
    device const float4* src [[buffer(0)]],
    device float4*       dst [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id] + float4(1.0f);
}

// ============================================================================
// 2. HARDWARE PROBE: Peak FP16 Compute Roofline (Pure FMA Saturation)
// ============================================================================
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
// 3. NAIVE REFERENCE KERNEL (Un-tiled 1-thread-per-element)
// ============================================================================
kernel void naive_q4_0_gemm(
    device const half*         A [[buffer(0)]], // [M x K]
    device const block_q4_0*   B [[buffer(1)]], // [N x (K / 32)]
    device half*               C [[buffer(2)]], // [M x N]
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
// 4. LLAMA.CPP STYLE mul_mm PRODUCTION BASELINE
// Tile size per threadgroup: 64 rows (M) x 32 cols (N)
// Threadgroup size: 64 threads (2 SIMDgroups of 32 threads)
// ============================================================================
#define TG_M 64
#define TG_N 32
#define BK   32

kernel void llamacpp_style_mul_mm_q4_0(
    device const half*         A [[buffer(0)]], // [M x K]
    device const block_q4_0*   B [[buffer(1)]], // [N x (K / 32)]
    device half*               C [[buffer(2)]], // [M x N]
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [TG_M x BK] = 64 * 32 * 2 = 4KB
    threadgroup half*          sh_B [[threadgroup(1)]], // [BK x TG_N] = 32 * 32 * 2 = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * TG_M;
    uint tg_col_start = tg_id.x * TG_N;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // SIMD 0: rows 0..31, SIMD 1: rows 32..63
    // Within SIMDgroup (32 threads): 4 rows x 8 cols of threads
    // Each thread computes 8 rows x 4 cols = 32 output elements
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
        // --- 1. Cooperative Load of A into Threadgroup Memory ---
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

        // --- 2. Cooperative Load & Dequantize of B into Threadgroup Memory ---
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

        // --- 3. Compute Outer Products from Drawer into Register Accumulators ---
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

    // --- 4. Store Accumulated Results to Output Matrix C ---
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
