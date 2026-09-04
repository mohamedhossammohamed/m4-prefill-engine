#include <metal_stdlib>
#include "include/metal/common/types.metal"
#include "include/metal/common/lsu.metal"
#include "include/metal/common/simd_reduce.metal"
#include "include/metal/common/sram_tile.metal"
#include "include/metal/common/math.metal"

#include "include/metal/quant/codec_traits.metal"
#include "include/metal/quant/q4_0.metal"
#include "include/metal/quant/mlx_4bit.metal"
#include "include/metal/quant/q4_k.metal"
#include "include/metal/quant/ternary_1_58.metal"
#include "include/metal/quant/var_rate_affine.metal"
#include "include/metal/quant/exl3.metal"
#include "include/metal/quant/prism_q2_0.metal"

#include "include/metal/ops/gemm_mma.metal"
#include "include/metal/ops/gemm_ternary_vec.metal"
#include "include/metal/ops/swiglu_dual_simd.metal"
#include "include/metal/ops/flash_attention.metal"

using namespace metal_llm;

kernel void test_modular_q4_0_gemm(
    device const half*              A [[buffer(0)]],
    device const quant::block_q4_0* B [[buffer(1)]],
    device half*                    C [[buffer(2)]],
    constant uint&                  M [[buffer(3)]],
    constant uint&                  N [[buffer(4)]],
    constant uint&                  K [[buffer(5)]],
    threadgroup half*               shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecQ4_0, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_mlx_4bit_gemm(
    device const half*                 A [[buffer(0)]],
    device const quant::block_mlx_4bit* B [[buffer(1)]],
    device half*                       C [[buffer(2)]],
    constant uint&                     M [[buffer(3)]],
    constant uint&                     N [[buffer(4)]],
    constant uint&                     K [[buffer(5)]],
    threadgroup half*                  shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecMLX4Bit, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_q4_k_gemm(
    device const half*              A [[buffer(0)]],
    device const quant::block_q4_K* B [[buffer(1)]],
    device half*                    C [[buffer(2)]],
    constant uint&                  M [[buffer(3)]],
    constant uint&                  N [[buffer(4)]],
    constant uint&                  K [[buffer(5)]],
    threadgroup half*               shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecQ4_K, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_ternary_1_58_gemm(
    device const half*                    A [[buffer(0)]],
    device const quant::block_ternary_1_58* B [[buffer(1)]],
    device half*                          C [[buffer(2)]],
    constant uint&                        M [[buffer(3)]],
    constant uint&                        N [[buffer(4)]],
    constant uint&                        K [[buffer(5)]],
    threadgroup half*                     shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecTernary158, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_ternary_1_58_vec_gemm(
    device const half*                    A [[buffer(0)]],
    device const quant::block_ternary_1_58* B [[buffer(1)]],
    device half*                          C [[buffer(2)]],
    constant uint&                        M [[buffer(3)]],
    constant uint&                        N [[buffer(4)]],
    constant uint&                        K [[buffer(5)]],
    threadgroup half*                     shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    ops::gemm_ternary_1_58_vec_core<false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id);
}

kernel void test_modular_var_rate_affine_gemm(
    device const half*                        A [[buffer(0)]],
    device const quant::block_var_rate_affine* B [[buffer(1)]],
    device half*                              C [[buffer(2)]],
    constant uint&                            M [[buffer(3)]],
    constant uint&                            N [[buffer(4)]],
    constant uint&                            K [[buffer(5)]],
    threadgroup half*                         shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecVarRateAffine, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_exl3_gemm(
    device const half*              A [[buffer(0)]],
    device const quant::block_exl3* B [[buffer(1)]],
    device half*                    C [[buffer(2)]],
    constant uint&                  M [[buffer(3)]],
    constant uint&                  N [[buffer(4)]],
    constant uint&                  K [[buffer(5)]],
    threadgroup half*               shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecEXL3, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_swiglu_q4_0(
    device const half*              A [[buffer(0)]],
    device const quant::block_q4_0* B_gate [[buffer(1)]],
    device const quant::block_q4_0* B_up [[buffer(2)]],
    device half*                    Out [[buffer(3)]],
    constant uint&                  M [[buffer(4)]],
    constant uint&                  N_mlp [[buffer(5)]],
    constant uint&                  K [[buffer(6)]],
    threadgroup half*               shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::swiglu_mma_dual_simd_core<quant::CodecQ4_0>(
        A, B_gate, B_up, Out, M, N_mlp, K, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_flash_attn_d64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  shmem [[threadgroup(0)]],
    uint3 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::flash_attn_mma_64x64_fp16_core<64>(
        Q, K, V, O, M, scale, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_flash_attn_d128(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  shmem [[threadgroup(0)]],
    uint3 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::flash_attn_mma_64x64_fp16_core<128>(
        Q, K, V, O, M, scale, shmem, tg_id, simd_lane_id, simd_group_id);
}

kernel void test_modular_prism_q2_0_gemm(
    device const half*                    A [[buffer(0)]],
    device const quant::block_prism_q2_0* B [[buffer(1)]],
    device half*                          C [[buffer(2)]],
    constant uint&                        M [[buffer(3)]],
    constant uint&                        N [[buffer(4)]],
    constant uint&                        K [[buffer(5)]],
    threadgroup half*                     shmem [[threadgroup(0)]],
    uint2 tg_id        [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    ops::block_mma_64x64_gemm_core<quant::CodecPrismQ2_0, false>(
        A, B, C, M, N, K, 0, 0, shmem, tg_id, simd_lane_id, simd_group_id);
}

