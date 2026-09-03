#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "../common/simd_reduce.metal"

namespace metal_llm {
namespace ops {

// Barrier-Free FlashAttention Core parameterized by Head Dimension D (64 or 128)
template <uint D>
inline void flash_attn_mma_64x64_fp16_core(
    device const half* Q,
    device const half* K,
    device const half* V,
    device half*       O,
    uint M,
    float scale,
    threadgroup half*  shmem,
    uint3 tg_id,
    uint  simd_lane_id,
    uint  simd_group_id)
{
    constexpr uint BQ = 64;
    constexpr uint BK = 64;
    constexpr uint D_FRAGS = D / 8;
    constexpr uint ELEMS_PER_THREAD = D / 32;

    constexpr uint STRIDE_Q      = D;
    constexpr uint STRIDE_K_T    = 64;
    constexpr uint STRIDE_V      = D;
    constexpr uint STRIDE_S_P    = 64;
    constexpr uint STRIDE_O_TILE = D;

    uint b_q = tg_id.x;
    uint h   = tg_id.y;
    uint q_start = b_q * BQ;

    if (q_start >= M) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint sg_row_start = simd_group_id * 16;

    threadgroup half* sh_Q      = shmem;
    threadgroup half* sh_K_T    = shmem + 64 * D;
    threadgroup half* sh_V      = shmem + 64 * D + D * 64;
    threadgroup half* sh_S_P    = sh_Q;
    threadgroup half* sh_O_tile = sh_K_T;

    float m_prev[16];
    float l_prev[16];
    half  o_acc[16][ELEMS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        m_prev[r] = -1e30f;
        l_prev[r] = 0.0f;
        #pragma unroll
        for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
            o_acc[r][e] = 0.0h;
        }
    }

    // 1. Cooperative Load of Q Tile [64, D] into sh_Q
    constexpr uint total_q_half4 = (64 * D) / 4;
    #pragma unroll
    for (uint idx = linear_tid; idx < total_q_half4; idx += 128) {
        uint r = idx / (D / 4);
        uint c_vec = idx % (D / 4);
        uint global_tok = q_start + r;
        half4 val = half4(0.0h);
        if (global_tok < M) {
            val = *reinterpret_cast<device const half4*>(Q + (h * M + global_tok) * D + c_vec * 4);
        }
        *reinterpret_cast<threadgroup half4*>(sh_Q + r * D + c_vec * 4) = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Loop Over Key/Value Tiles with Causal Triangular Masking
    uint num_k_tiles = (M + BK - 1) / BK;
    uint max_causal_k_tile = (min((b_q + 1) * BQ, M) - 1) / BK;
    uint loop_k_tiles = min(max_causal_k_tile + 1, num_k_tiles);

    for (uint b_k = 0; b_k < loop_k_tiles; b_k++) {
        uint k_start = b_k * BK;

        // Cooperative Load of K and V Tiles
        constexpr uint total_kv_half4 = (64 * D) / 4;
        #pragma unroll
        for (uint idx = linear_tid; idx < total_kv_half4; idx += 128) {
            uint k_row = idx / (D / 4);
            uint d_vec = idx % (D / 4);
            uint global_k_tok = k_start + k_row;

            half4 k_val = half4(0.0h);
            half4 v_val = half4(0.0h);
            if (global_k_tok < M) {
                k_val = *reinterpret_cast<device const half4*>(K + (h * M + global_k_tok) * D + d_vec * 4);
                v_val = *reinterpret_cast<device const half4*>(V + (h * M + global_k_tok) * D + d_vec * 4);
            }

            // Transpose K on-the-fly into sh_K_T [D, 64]
            sh_K_T[(d_vec * 4 + 0) * 64 + k_row] = k_val.x;
            sh_K_T[(d_vec * 4 + 1) * 64 + k_row] = k_val.y;
            sh_K_T[(d_vec * 4 + 2) * 64 + k_row] = k_val.z;
            sh_K_T[(d_vec * 4 + 3) * 64 + k_row] = k_val.w;

            // Store V into sh_V [64, D]
            *reinterpret_cast<threadgroup half4*>(sh_V + k_row * D + d_vec * 4) = v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hardware Tensor-Core Matrix Multiply: S_sg = Q_sg * K^T [16, 64]
        simdgroup_matrix<half, 8, 8> s_frag[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                s_frag[r][c] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        #pragma unroll
        for (uint kd = 0; kd < D_FRAGS; kd++) {
            simdgroup_matrix<half, 8, 8> q_frag[2];
            simdgroup_load(q_frag[0], sh_Q + (sg_row_start + 0) * STRIDE_Q + kd * 8, STRIDE_Q);
            simdgroup_load(q_frag[1], sh_Q + (sg_row_start + 8) * STRIDE_Q + kd * 8, STRIDE_Q);

            simdgroup_matrix<half, 8, 8> kt_frag[8];
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                simdgroup_load(kt_frag[c], sh_K_T + (kd * 8) * STRIDE_K_T + (c * 8), STRIDE_K_T);
            }

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    simdgroup_multiply_accumulate(s_frag[r][c], q_frag[r], kt_frag[c], s_frag[r][c]);
                }
            }
        }

        // Store S_sg to shared memory
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                simdgroup_store(s_frag[r][c], sh_S_P + (sg_row_start + r * 8) * STRIDE_S_P + (c * 8), STRIDE_S_P);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Causal Masking & Online Softmax Reduction using simd_shuffle_down
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            uint global_q_tok = q_start + sg_row_start + r;
            uint c_a = simd_lane_id;
            uint c_b = simd_lane_id + 32;
            uint global_k_tok_a = k_start + c_a;
            uint global_k_tok_b = k_start + c_b;

            float s_a = -1e30f;
            if (global_q_tok < M && global_k_tok_a <= global_q_tok && global_k_tok_a < M) {
                s_a = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] * scale;
            }

            float s_b = -1e30f;
            if (global_q_tok < M && global_k_tok_b <= global_q_tok && global_k_tok_b < M) {
                s_b = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] * scale;
            }

            // SIMD reduction for row max
            float block_max = max(s_a, s_b);
            block_max = common::simd_reduce_max(block_max);

            float new_max = max(m_prev[r], block_max);
            float alpha = (m_prev[r] > -1e20f) ? fast::exp(m_prev[r] - new_max) : 0.0f;
            float beta = (block_max > -1e20f) ? fast::exp(block_max - new_max) : 0.0f;

            float p_a = (s_a > -1e20f) ? (fast::exp(s_a - block_max) * beta) : 0.0f;
            float p_b = (s_b > -1e20f) ? (fast::exp(s_b - block_max) * beta) : 0.0f;

            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] = (half)p_a;
            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] = (half)p_b;

            // SIMD reduction for row sum
            float block_sum = p_a + p_b;
            block_sum = common::simd_reduce_sum(block_sum);

            float new_sum = l_prev[r] * alpha + block_sum;
            m_prev[r] = new_max;
            l_prev[r] = new_sum;

            // Rescale output accumulators for this row
            half alpha_h = (half)alpha;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = o_acc[r][e] * alpha_h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hardware Tensor-Core Matrix Multiply: O_tile = P_sg * V [16, D]
        simdgroup_matrix<half, 8, 8> o_frag[2][D_FRAGS];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (uint d = 0; d < D_FRAGS; d++) {
                o_frag[r][d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
        }

        #pragma unroll
        for (uint kv = 0; kv < 8; kv++) {
            simdgroup_matrix<half, 8, 8> p_frag[2];
            simdgroup_load(p_frag[0], sh_S_P + (sg_row_start + 0) * STRIDE_S_P + (kv * 8), STRIDE_S_P);
            simdgroup_load(p_frag[1], sh_S_P + (sg_row_start + 8) * STRIDE_S_P + (kv * 8), STRIDE_S_P);

            simdgroup_matrix<half, 8, 8> v_frag[D_FRAGS];
            #pragma unroll
            for (uint d = 0; d < D_FRAGS; d++) {
                simdgroup_load(v_frag[d], sh_V + (kv * 8) * STRIDE_V + (d * 8), STRIDE_V);
            }

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < D_FRAGS; d++) {
                    simdgroup_multiply_accumulate(o_frag[r][d], p_frag[r], v_frag[d], o_frag[r][d]);
                }
            }
        }

        // Store O_tile to shared memory (reusing sh_K_T after barrier)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (uint d = 0; d < D_FRAGS; d++) {
                simdgroup_store(o_frag[r][d], sh_O_tile + (sg_row_start + r * 8) * STRIDE_O_TILE + (d * 8), STRIDE_O_TILE);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate O_tile into registers
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                uint col = simd_lane_id * ELEMS_PER_THREAD + e;
                o_acc[r][e] += sh_O_tile[(sg_row_start + r) * STRIDE_O_TILE + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Epilogue: Write normalized output O to global memory
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_q_tok = q_start + sg_row_start + r;
        if (global_q_tok < M) {
            half inv_sum = (l_prev[r] > 0.0f) ? (half)(1.0f / l_prev[r]) : 0.0h;
            device half* out_ptr = O + (h * M + global_q_tok) * D + simd_lane_id * ELEMS_PER_THREAD;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                out_ptr[e] = o_acc[r][e] * inv_sum;
            }
        }
    }
}

} // namespace ops
} // namespace metal_llm
