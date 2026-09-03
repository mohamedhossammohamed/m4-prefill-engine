#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "../common/math.metal"

namespace metal_llm {
namespace ops {

template <typename TCodec>
inline void swiglu_mma_dual_simd_core(
    device const half*                        A,
    device const typename TCodec::BlockType*  B_gate,
    device const typename TCodec::BlockType*  B_up,
    device half*                              Out,
    uint M,
    uint N_mlp,
    uint K,
    threadgroup half*                         shmem,
    uint2 tg_id,
    uint  simd_lane_id,
    uint  simd_group_id)
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Buffer size for A: [2][64][36] = 4608 halfs
    // Buffer size for B_gate: [2][32][32] = 2048 halfs
    // Buffer size for B_up:   [2][32][32] = 2048 halfs
    // Total shared memory = 8704 halfs = 17408 bytes
    threadgroup half (*sh_A)[64][36]      = (threadgroup half (*)[64][36])shmem;
    threadgroup half (*sh_B_gate)[32][32] = (threadgroup half (*)[32][32])(shmem + 4608);
    threadgroup half (*sh_B_up)[32][32]   = (threadgroup half (*)[32][32])(shmem + 4608 + 2048);

    uint sg_row_offset = (simd_group_id & 1) * 32;
    bool is_gate_sg = (simd_group_id < 2);

    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint num_k_blocks = K / 32;

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

            if (is_gate_sg) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_load(b_frag[c], &sh_B_gate[buf_idx][k_off][c * 8], 32);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_load(b_frag[c], &sh_B_up[buf_idx][k_off][c * 8], 32);
                }
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

    // Helper macro/lambda for load A
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        uint idx = linear_tid * 2 + i;
        uint r = idx / 4;
        uint c = (idx % 4) * 8;
        uint global_r = tg_row_start + r;
        uint global_c = 0 * 32 + c;
        float4 val = float4(0.0f);
        if (global_r < M && global_c < K) {
            val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
        }
        *reinterpret_cast<threadgroup float2*>(&sh_A[0][r][c])     = val.xy;
        *reinterpret_cast<threadgroup float2*>(&sh_A[0][r][c + 4]) = val.zw;
    }

    if (linear_tid < 32) {
        uint col = tg_col_start + linear_tid;
        if (col < N_mlp && 0 < num_k_blocks) {
            threadgroup half* dst_col = (threadgroup half*)sh_B_gate[0];
            TCodec::unpack_column(B_gate, col, 0, K, dst_col, linear_tid);
        } else {
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                sh_B_gate[0][k][linear_tid] = 0.0h;
            }
        }
    } else if (linear_tid >= 32 && linear_tid < 64) {
        uint up_col_idx = linear_tid - 32;
        uint col = tg_col_start + up_col_idx;
        if (col < N_mlp && 0 < num_k_blocks) {
            threadgroup half* dst_col = (threadgroup half*)sh_B_up[0];
            TCodec::unpack_column(B_up, col, 0, K, dst_col, up_col_idx);
        } else {
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                sh_B_up[0][k][up_col_idx] = 0.0h;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Double-buffered main loop
    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur = kb & 1;
        uint nxt = cur ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                uint idx = linear_tid * 2 + i;
                uint r = idx / 4;
                uint c = (idx % 4) * 8;
                uint global_r = tg_row_start + r;
                uint global_c = next_kb * 32 + c;
                float4 val = float4(0.0f);
                if (global_r < M && global_c < K) {
                    val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
                }
                *reinterpret_cast<threadgroup float2*>(&sh_A[nxt][r][c])     = val.xy;
                *reinterpret_cast<threadgroup float2*>(&sh_A[nxt][r][c + 4]) = val.zw;
            }

            if (linear_tid < 32) {
                uint col = tg_col_start + linear_tid;
                if (col < N_mlp) {
                    threadgroup half* dst_col = (threadgroup half*)sh_B_gate[nxt];
                    TCodec::unpack_column(B_gate, col, next_kb, K, dst_col, linear_tid);
                } else {
                    #pragma unroll
                    for (int k = 0; k < 32; k++) {
                        sh_B_gate[nxt][k][linear_tid] = 0.0h;
                    }
                }
            } else if (linear_tid >= 32 && linear_tid < 64) {
                uint up_col_idx = linear_tid - 32;
                uint col = tg_col_start + up_col_idx;
                if (col < N_mlp) {
                    threadgroup half* dst_col = (threadgroup half*)sh_B_up[nxt];
                    TCodec::unpack_column(B_up, col, next_kb, K, dst_col, up_col_idx);
                } else {
                    #pragma unroll
                    for (int k = 0; k < 32; k++) {
                        sh_B_up[nxt][k][up_col_idx] = 0.0h;
                    }
                }
            }
        }

        compute_mma(cur);

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Epilogue: Write MMA fragments to threadgroup SRAM, then compute SiLU(Gate) * Up in SRAM
    threadgroup float (*sh_Gate)[32] = (threadgroup float (*)[32])shmem;
    threadgroup float (*sh_Up)[32]   = (threadgroup float (*)[32])(shmem + 4096);

    if (simd_group_id == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Gate[0 + r * 8][c * 8], 32);
            }
        }
    } else if (simd_group_id == 1) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Gate[32 + r * 8][c * 8], 32);
            }
        }
    } else if (simd_group_id == 2) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Up[0 + r * 8][c * 8], 32);
            }
        }
    } else if (simd_group_id == 3) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Up[32 + r * 8][c * 8], 32);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fused On-Chip SwiGLU Activation & DRAM Writeback
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint elem_idx = linear_tid * 16 + i;
        uint r = elem_idx / 32;
        uint c = elem_idx % 32;
        uint global_r = tg_row_start + r;
        uint global_c = tg_col_start + c;
        if (global_r < M && global_c < N_mlp) {
            float g = sh_Gate[r][c];
            float u = sh_Up[r][c];
            float silu_g = g / (1.0f + exp(-g));
            Out[global_r * N_mlp + global_c] = (half)(silu_g * u);
        }
    }
}

} // namespace ops
} // namespace metal_llm
