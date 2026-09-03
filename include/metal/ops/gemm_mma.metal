#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"

namespace metal_llm {
namespace ops {

template <typename TCodec, bool DIRECT_HEAD_ROUTING = false>
inline void block_mma_64x64_gemm_core(
    device const half*                         A,
    device const typename TCodec::BlockType*   B,
    device half*                               C,
    uint M,
    uint N,
    uint K,
    uint H,
    uint D,
    threadgroup half*                          shmem,
    uint2 tg_id,
    uint  simd_lane_id,
    uint  simd_group_id)
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Shared memory layout (matches monolithic quant_router_kernels.metal stride-32 layout):
    // sh_A: [64][32] = 2048 halfs (4KB)
    // sh_B: [32][64] = 2048 halfs (4KB)
    threadgroup half* sh_A = shmem;
    threadgroup half* sh_B = shmem + 2048;

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

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        // 1. Cooperative coalesced loading of activation A (64 rows x 32 cols = 256 float4s)
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

        // 2. Pluggable Codec Dequantization of Weight B (64 columns)
        if (linear_tid < 64) {
            uint b_col_idx = tg_col_start + linear_tid;
            if (b_col_idx < N) {
                TCodec::unpack_column(B, b_col_idx, kb, K, sh_B, linear_tid);
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
                uint h = global_c / D;
                uint d_idx = global_c % D;
                C[(h * M + global_r) * D + d_idx] = (half)sh_Out[elem_idx];
            } else {
                C[global_r * N + global_c] = (half)sh_Out[elem_idx];
            }
        }
    }
}

} // namespace ops
} // namespace metal_llm
