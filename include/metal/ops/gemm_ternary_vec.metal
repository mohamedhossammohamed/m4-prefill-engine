#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "../quant/ternary_1_58.metal"

namespace metal_llm {
namespace ops {

template <bool DIRECT_HEAD_ROUTING = false>
inline void gemm_ternary_1_58_vec_core(
    device const half*                          A,
    device const quant::block_ternary_1_58*     B,
    device half*                                C,
    uint M,
    uint N,
    uint K,
    uint H,
    uint D,
    threadgroup half*                           shmem,
    uint2 tg_id,
    uint  simd_lane_id)
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

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint idx = simd_lane_id * 4 + i;
        uint r = idx / 4;
        uint c = (idx % 4) * 8;
        uint global_r = tg_row_start + r;
        uint global_c = 0 * 32 + c;
        float4 val = float4(0.0f);
        if (global_r < M && global_c < K) {
            val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
        }
        *reinterpret_cast<threadgroup float4*>(&sh_A[0][r][c]) = val;
    }

    quant::block_ternary_1_58 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        quant::block_ternary_1_58 q_next;

        if (next_kb < num_k_blocks) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint idx = simd_lane_id * 4 + i;
                uint r = idx / 4;
                uint c = (idx % 4) * 8;
                uint global_r = tg_row_start + r;
                uint global_c = next_kb * 32 + c;
                float4 val = float4(0.0f);
                if (global_r < M && global_c < K) {
                    val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
                }
                *reinterpret_cast<threadgroup float4*>(&sh_A[nxt_buf][r][c]) = val;
            }
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
                if (DIRECT_HEAD_ROUTING && H > 0 && D > 0) {
                    uint h = col_idx / D;
                    uint d_idx = col_idx % D;
                    C[(h * M + global_r) * D + d_idx] = (half)acc[r];
                } else {
                    C[global_r * N + col_idx] = (half)acc[r];
                }
            }
        }
    }
}

} // namespace ops
} // namespace metal_llm
