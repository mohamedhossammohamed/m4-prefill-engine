#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct block_q8_0 {
    half d;          // Scale factor (2 bytes)
    int8_t qs[32];   // 32 quantized 8-bit values (32 bytes)
};

// ============================================================================
// 0. COLD-CACHE EVICTION KERNEL (32MB Read/Write to flush Apple M4 24MB SLC)
// ============================================================================
kernel void cold_cache_evict_kernel(
    device uint* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        uint val = data[tid];
        val = (val ^ 0x5A5A5A5A) + tid;
        data[tid] = val;
    }
}

// ============================================================================
// 1. IN-RAM FUSED 2D BLOCK-MMA FLASHATTENTION (FP16 & Q8_0)
// ============================================================================
template<uint D>
inline void flash_attn_mma_64x64_fp16_impl(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  smem_raw [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BQ = 64;
    constexpr uint BK = 64;
    constexpr uint D_FRAGS = D / 8;     // 8 for D=64, 16 for D=128
    constexpr uint NUM_PASSES = D_FRAGS / 8; // 1 for D=64, 2 for D=128
    constexpr uint STRIDE_Q = D;
    constexpr uint STRIDE_K_T = 64;
    constexpr uint STRIDE_V = D;
    constexpr uint STRIDE_S_P = 64;
    constexpr uint STRIDE_O_TILE = D;

    uint b_q = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    uint q_start = b_q * BQ;

    if (q_start >= M) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Shared Memory Layout:
    constexpr uint Q_SIZE = 64 * D;
    constexpr uint K_T_SIZE = D * 64;
    constexpr uint V_SIZE = 64 * D;

    threadgroup half* sh_Q      = smem_raw;
    threadgroup half* sh_K_T    = smem_raw + Q_SIZE;
    threadgroup half* sh_V      = smem_raw + Q_SIZE + K_T_SIZE;
    threadgroup half* sh_S_P    = smem_raw + Q_SIZE + K_T_SIZE + V_SIZE;
    threadgroup half* sh_O_tile = sh_K_T; // Reuse sh_K_T memory space after S is computed!

    // SIMDgroup Row Partitioning (4 SIMDgroups, 16 rows each)
    uint sg_row_start = simd_group_id * 16;

    // Register State for Online Softmax per SIMDgroup (16 rows)
    float m_prev[16];
    float l_prev[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        m_prev[r] = -1e30f;
        l_prev[r] = 0.0f;
    }

    // Output Accumulators in Registers
    constexpr uint ELEMS_PER_THREAD = D / 32; // 2 for D=64, 4 for D=128
    half o_acc[16][ELEMS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
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

        // A. Cooperative Load of K and V Tiles
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

        // B. Hardware Tensor-Core Matrix Multiply: S_sg = Q_sg * K^T [16, 64]
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

        // C. Causal Masking & Online Softmax Reduction using simd_shuffle_down
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
            block_max = max(block_max, simd_shuffle_down(block_max, 16));
            block_max = max(block_max, simd_shuffle_down(block_max, 8));
            block_max = max(block_max, simd_shuffle_down(block_max, 4));
            block_max = max(block_max, simd_shuffle_down(block_max, 2));
            block_max = max(block_max, simd_shuffle_down(block_max, 1));
            block_max = simd_broadcast(block_max, 0);

            float new_max = max(m_prev[r], block_max);
            float alpha = (m_prev[r] > -1e20f) ? metal::fast::exp(m_prev[r] - new_max) : 0.0f;
            float beta = (block_max > -1e20f) ? metal::fast::exp(block_max - new_max) : 0.0f;

            float p_a = (s_a > -1e20f) ? (metal::fast::exp(s_a - block_max) * beta) : 0.0f;
            float p_b = (s_b > -1e20f) ? (metal::fast::exp(s_b - block_max) * beta) : 0.0f;

            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] = (half)p_a;
            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] = (half)p_b;

            // SIMD reduction for row sum
            float block_sum = p_a + p_b;
            block_sum += simd_shuffle_down(block_sum, 16);
            block_sum += simd_shuffle_down(block_sum, 8);
            block_sum += simd_shuffle_down(block_sum, 4);
            block_sum += simd_shuffle_down(block_sum, 2);
            block_sum += simd_shuffle_down(block_sum, 1);
            block_sum = simd_broadcast(block_sum, 0);

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

        // D. Hardware Tensor-Core Matrix Multiply: O_tile = P_sg * V [16, D] in 8-fragment passes
        #pragma unroll
        for (uint pass = 0; pass < NUM_PASSES; pass++) {
            simdgroup_matrix<half, 8, 8> o_frag[2][8];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    o_frag[r][d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            #pragma unroll
            for (uint kv = 0; kv < 8; kv++) {
                simdgroup_matrix<half, 8, 8> p_frag[2];
                simdgroup_load(p_frag[0], sh_S_P + (sg_row_start + 0) * STRIDE_S_P + (kv * 8), STRIDE_S_P);
                simdgroup_load(p_frag[1], sh_S_P + (sg_row_start + 8) * STRIDE_S_P + (kv * 8), STRIDE_S_P);

                simdgroup_matrix<half, 8, 8> v_frag[8];
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_load(v_frag[d], sh_V + (kv * 8) * STRIDE_V + (pass * 64 + d * 8), STRIDE_V);
                }

                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    #pragma unroll
                    for (uint d = 0; d < 8; d++) {
                        simdgroup_multiply_accumulate(o_frag[r][d], p_frag[r], v_frag[d], o_frag[r][d]);
                    }
                }
            }

            // Store O_tile to shared memory (reusing sh_K_T after barrier)
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_store(o_frag[r][d], sh_O_tile + (sg_row_start + r * 8) * STRIDE_O_TILE + (pass * 64 + d * 8), STRIDE_O_TILE);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // E. Accumulate O_tile into registers
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

template<uint D>
inline void flash_attn_mma_64x64_q8_0_impl(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    threadgroup half*          smem_raw [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BQ = 64;
    constexpr uint BK = 64;
    constexpr uint D_FRAGS = D / 8;     // 8 for D=64, 16 for D=128
    constexpr uint NUM_PASSES = D_FRAGS / 8;
    constexpr uint BLOCKS_PER_TOKEN = D / 32; // 2 for D=64, 4 for D=128
    constexpr uint TOTAL_KV_BLOCKS = 64 * BLOCKS_PER_TOKEN; // 128 for D=64, 256 for D=128
    constexpr uint BLOCKS_PER_THREAD = TOTAL_KV_BLOCKS / 128; // 1 for D=64, 2 for D=128

    constexpr uint STRIDE_Q = D;
    constexpr uint STRIDE_K_T = 64;
    constexpr uint STRIDE_V = D;
    constexpr uint STRIDE_S_P = 64;
    constexpr uint STRIDE_O_TILE = D;

    uint b_q = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    uint q_start = b_q * BQ;

    if (q_start >= M) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Shared Memory Layout:
    constexpr uint Q_SIZE = 64 * D;
    constexpr uint K_T_SIZE = D * 64;
    constexpr uint V_SIZE = 64 * D;

    threadgroup half* sh_Q      = smem_raw;
    threadgroup half* sh_K_T    = smem_raw + Q_SIZE;
    threadgroup half* sh_V      = smem_raw + Q_SIZE + K_T_SIZE;
    threadgroup half* sh_S_P    = smem_raw + Q_SIZE + K_T_SIZE + V_SIZE;
    threadgroup half* sh_O_tile = sh_K_T;

    uint sg_row_start = simd_group_id * 16;

    float m_prev[16];
    float l_prev[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        m_prev[r] = -1e30f;
        l_prev[r] = 0.0f;
    }

    constexpr uint ELEMS_PER_THREAD = D / 32;
    half o_acc[16][ELEMS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        #pragma unroll
        for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
            o_acc[r][e] = 0.0h;
        }
    }

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

    uint num_k_tiles = (M + BK - 1) / BK;
    uint max_causal_k_tile = (min((b_q + 1) * BQ, M) - 1) / BK;
    uint loop_k_tiles = min(max_causal_k_tile + 1, num_k_tiles);

    for (uint b_k = 0; b_k < loop_k_tiles; b_k++) {
        uint k_start = b_k * BK;

        // Dynamic On-The-Fly Q8_0 Dequantization into Threadgroup SRAM
        #pragma unroll
        for (uint b = 0; b < BLOCKS_PER_THREAD; b++) {
            uint blk_local_idx = linear_tid * BLOCKS_PER_THREAD + b;
            uint k_row = blk_local_idx / BLOCKS_PER_TOKEN;
            uint sub_blk = blk_local_idx % BLOCKS_PER_TOKEN;
            uint global_k_tok = k_start + k_row;

            if (global_k_tok < M) {
                uint global_blk_idx = (h * M + global_k_tok) * BLOCKS_PER_TOKEN + sub_blk;
                block_q8_0 k_blk = K_q8[global_blk_idx];
                block_q8_0 v_blk = V_q8[global_blk_idx];

                half kd = k_blk.d;
                half vd = v_blk.d;

                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    half k_val = (half)k_blk.qs[i] * kd;
                    half v_val = (half)v_blk.qs[i] * vd;

                    sh_K_T[(sub_blk * 32 + i) * 64 + k_row] = k_val;
                    sh_V[k_row * D + sub_blk * 32 + i] = v_val;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    sh_K_T[(sub_blk * 32 + i) * 64 + k_row] = 0.0h;
                    sh_V[k_row * D + sub_blk * 32 + i] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                simdgroup_store(s_frag[r][c], sh_S_P + (sg_row_start + r * 8) * STRIDE_S_P + (c * 8), STRIDE_S_P);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

            float block_max = max(s_a, s_b);
            block_max = max(block_max, simd_shuffle_down(block_max, 16));
            block_max = max(block_max, simd_shuffle_down(block_max, 8));
            block_max = max(block_max, simd_shuffle_down(block_max, 4));
            block_max = max(block_max, simd_shuffle_down(block_max, 2));
            block_max = max(block_max, simd_shuffle_down(block_max, 1));
            block_max = simd_broadcast(block_max, 0);

            float new_max = max(m_prev[r], block_max);
            float alpha = (m_prev[r] > -1e20f) ? metal::fast::exp(m_prev[r] - new_max) : 0.0f;
            float beta = (block_max > -1e20f) ? metal::fast::exp(block_max - new_max) : 0.0f;

            float p_a = (s_a > -1e20f) ? (metal::fast::exp(s_a - block_max) * beta) : 0.0f;
            float p_b = (s_b > -1e20f) ? (metal::fast::exp(s_b - block_max) * beta) : 0.0f;

            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] = (half)p_a;
            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] = (half)p_b;

            float block_sum = p_a + p_b;
            block_sum += simd_shuffle_down(block_sum, 16);
            block_sum += simd_shuffle_down(block_sum, 8);
            block_sum += simd_shuffle_down(block_sum, 4);
            block_sum += simd_shuffle_down(block_sum, 2);
            block_sum += simd_shuffle_down(block_sum, 1);
            block_sum = simd_broadcast(block_sum, 0);

            float new_sum = l_prev[r] * alpha + block_sum;
            m_prev[r] = new_max;
            l_prev[r] = new_sum;

            half alpha_h = (half)alpha;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = o_acc[r][e] * alpha_h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint pass = 0; pass < NUM_PASSES; pass++) {
            simdgroup_matrix<half, 8, 8> o_frag[2][8];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    o_frag[r][d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            #pragma unroll
            for (uint kv = 0; kv < 8; kv++) {
                simdgroup_matrix<half, 8, 8> p_frag[2];
                simdgroup_load(p_frag[0], sh_S_P + (sg_row_start + 0) * STRIDE_S_P + (kv * 8), STRIDE_S_P);
                simdgroup_load(p_frag[1], sh_S_P + (sg_row_start + 8) * STRIDE_S_P + (kv * 8), STRIDE_S_P);

                simdgroup_matrix<half, 8, 8> v_frag[8];
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_load(v_frag[d], sh_V + (kv * 8) * STRIDE_V + (pass * 64 + d * 8), STRIDE_V);
                }

                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    #pragma unroll
                    for (uint d = 0; d < 8; d++) {
                        simdgroup_multiply_accumulate(o_frag[r][d], p_frag[r], v_frag[d], o_frag[r][d]);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_store(o_frag[r][d], sh_O_tile + (sg_row_start + r * 8) * STRIDE_O_TILE + (pass * 64 + d * 8), STRIDE_O_TILE);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

// In-RAM Kernel Entry Points
kernel void flash_attn_mma_64x64_fp16_d64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    flash_attn_mma_64x64_fp16_impl<64>(Q, K, V, O, M, scale, smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void flash_attn_mma_64x64_fp16_d128(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    flash_attn_mma_64x64_fp16_impl<128>(Q, K, V, O, M, scale, smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void flash_attn_mma_64x64_q8_0_d64(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    threadgroup half*          smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    flash_attn_mma_64x64_q8_0_impl<64>(Q, K_q8, V_q8, O, M, scale, smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void flash_attn_mma_64x64_q8_0_d128(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    threadgroup half*          smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    flash_attn_mma_64x64_q8_0_impl<128>(Q, K_q8, V_q8, O, M, scale, smem, tg_pos, simd_lane_id, simd_group_id);
}

// ============================================================================
// 2. CHUNKED OUT-OF-CORE STREAMING FLASHATTENTION WITH ONLINE SOFTMAX MERGE
// ============================================================================
template<uint D>
inline void streaming_flash_attn_chunk_fp16_impl(
    device const half* Q [[buffer(0)]],
    device const half* K_chunk [[buffer(1)]],
    device const half* V_chunk [[buffer(2)]],
    device half*       O [[buffer(3)]],
    device float*      m_state [[buffer(4)]],
    device float*      l_state [[buffer(5)]],
    device half*       O_state [[buffer(6)]],
    constant uint&     M [[buffer(7)]],
    constant uint&     C_slot [[buffer(8)]],
    constant uint&     k_chunk_start [[buffer(9)]],
    constant uint&     k_chunk_len [[buffer(10)]],
    constant uint&     is_first_chunk [[buffer(11)]],
    constant uint&     is_last_chunk [[buffer(12)]],
    constant float&    scale [[buffer(13)]],
    threadgroup half*  smem_raw [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BQ = 64;
    constexpr uint BK = 64;
    constexpr uint D_FRAGS = D / 8;     // 8 for D=64, 16 for D=128
    constexpr uint NUM_PASSES = D_FRAGS / 8;
    constexpr uint STRIDE_Q = D;
    constexpr uint STRIDE_K_T = 64;
    constexpr uint STRIDE_V = D;
    constexpr uint STRIDE_S_P = 64;
    constexpr uint STRIDE_O_TILE = D;

    uint b_q = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    uint q_start = b_q * BQ;

    if (q_start >= M) return;

    if (q_start + BQ <= k_chunk_start) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    constexpr uint Q_SIZE = 64 * D;
    constexpr uint K_T_SIZE = D * 64;
    constexpr uint V_SIZE = 64 * D;

    threadgroup half* sh_Q      = smem_raw;
    threadgroup half* sh_K_T    = smem_raw + Q_SIZE;
    threadgroup half* sh_V      = smem_raw + Q_SIZE + K_T_SIZE;
    threadgroup half* sh_S_P    = smem_raw + Q_SIZE + K_T_SIZE + V_SIZE;
    threadgroup half* sh_O_tile = sh_K_T;

    uint sg_row_start = simd_group_id * 16;

    float m_prev[16];
    float l_prev[16];
    constexpr uint ELEMS_PER_THREAD = D / 32;
    half o_acc[16][ELEMS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_q_tok = q_start + sg_row_start + r;
        if (is_first_chunk != 0) {
            m_prev[r] = -1e30f;
            l_prev[r] = 0.0f;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = 0.0h;
            }
        } else {
            if (global_q_tok < M && global_q_tok >= k_chunk_start) {
                m_prev[r] = m_state[h * M + global_q_tok];
                l_prev[r] = l_state[h * M + global_q_tok];
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    uint col = simd_lane_id * ELEMS_PER_THREAD + e;
                    o_acc[r][e] = O_state[(h * M + global_q_tok) * D + col];
                }
            } else {
                m_prev[r] = -1e30f;
                l_prev[r] = 0.0f;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    o_acc[r][e] = 0.0h;
                }
            }
        }
    }

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

    uint num_k_tiles = (k_chunk_len + BK - 1) / BK;
    uint max_causal_k_tile = (min((b_q + 1) * BQ, M) - 1 >= k_chunk_start) ? ((min((b_q + 1) * BQ, M) - 1 - k_chunk_start) / BK) : 0;
    uint loop_k_tiles = min(max_causal_k_tile + 1, num_k_tiles);

    for (uint b_k = 0; b_k < loop_k_tiles; b_k++) {
        uint k_tile_local = b_k * BK;
        uint k_tile_global = k_chunk_start + k_tile_local;

        constexpr uint total_kv_half4 = (64 * D) / 4;
        #pragma unroll
        for (uint idx = linear_tid; idx < total_kv_half4; idx += 128) {
            uint k_row = idx / (D / 4);
            uint d_vec = idx % (D / 4);
            uint k_local_tok = k_tile_local + k_row;
            uint global_k_tok = k_tile_global + k_row;

            half4 k_val = half4(0.0h);
            half4 v_val = half4(0.0h);
            if (global_k_tok < M && k_local_tok < k_chunk_len) {
                k_val = *reinterpret_cast<device const half4*>(K_chunk + (h * C_slot + k_local_tok) * D + d_vec * 4);
                v_val = *reinterpret_cast<device const half4*>(V_chunk + (h * C_slot + k_local_tok) * D + d_vec * 4);
            }

            sh_K_T[(d_vec * 4 + 0) * 64 + k_row] = k_val.x;
            sh_K_T[(d_vec * 4 + 1) * 64 + k_row] = k_val.y;
            sh_K_T[(d_vec * 4 + 2) * 64 + k_row] = k_val.z;
            sh_K_T[(d_vec * 4 + 3) * 64 + k_row] = k_val.w;

            *reinterpret_cast<threadgroup half4*>(sh_V + k_row * D + d_vec * 4) = v_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                simdgroup_store(s_frag[r][c], sh_S_P + (sg_row_start + r * 8) * STRIDE_S_P + (c * 8), STRIDE_S_P);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int r = 0; r < 16; r++) {
            uint global_q_tok = q_start + sg_row_start + r;
            uint c_a = simd_lane_id;
            uint c_b = simd_lane_id + 32;
            uint global_k_tok_a = k_chunk_start + k_tile_local + c_a;
            uint global_k_tok_b = k_chunk_start + k_tile_local + c_b;

            float s_a = -1e30f;
            if (global_q_tok < M && global_k_tok_a <= global_q_tok && global_k_tok_a < M && (k_tile_local + c_a) < k_chunk_len) {
                s_a = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] * scale;
            }

            float s_b = -1e30f;
            if (global_q_tok < M && global_k_tok_b <= global_q_tok && global_k_tok_b < M && (k_tile_local + c_b) < k_chunk_len) {
                s_b = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] * scale;
            }

            float block_max = max(s_a, s_b);
            block_max = max(block_max, simd_shuffle_down(block_max, 16));
            block_max = max(block_max, simd_shuffle_down(block_max, 8));
            block_max = max(block_max, simd_shuffle_down(block_max, 4));
            block_max = max(block_max, simd_shuffle_down(block_max, 2));
            block_max = max(block_max, simd_shuffle_down(block_max, 1));
            block_max = simd_broadcast(block_max, 0);

            float new_max = max(m_prev[r], block_max);
            float alpha = (m_prev[r] > -1e20f) ? metal::fast::exp(m_prev[r] - new_max) : 0.0f;
            float beta = (block_max > -1e20f) ? metal::fast::exp(block_max - new_max) : 0.0f;

            float p_a = (s_a > -1e20f) ? (metal::fast::exp(s_a - block_max) * beta) : 0.0f;
            float p_b = (s_b > -1e20f) ? (metal::fast::exp(s_b - block_max) * beta) : 0.0f;

            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] = (half)p_a;
            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] = (half)p_b;

            float block_sum = p_a + p_b;
            block_sum += simd_shuffle_down(block_sum, 16);
            block_sum += simd_shuffle_down(block_sum, 8);
            block_sum += simd_shuffle_down(block_sum, 4);
            block_sum += simd_shuffle_down(block_sum, 2);
            block_sum += simd_shuffle_down(block_sum, 1);
            block_sum = simd_broadcast(block_sum, 0);

            float new_sum = l_prev[r] * alpha + block_sum;
            m_prev[r] = new_max;
            l_prev[r] = new_sum;

            half alpha_h = (half)alpha;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = o_acc[r][e] * alpha_h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint pass = 0; pass < NUM_PASSES; pass++) {
            simdgroup_matrix<half, 8, 8> o_frag[2][8];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    o_frag[r][d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            #pragma unroll
            for (uint kv = 0; kv < 8; kv++) {
                simdgroup_matrix<half, 8, 8> p_frag[2];
                simdgroup_load(p_frag[0], sh_S_P + (sg_row_start + 0) * STRIDE_S_P + (kv * 8), STRIDE_S_P);
                simdgroup_load(p_frag[1], sh_S_P + (sg_row_start + 8) * STRIDE_S_P + (kv * 8), STRIDE_S_P);

                simdgroup_matrix<half, 8, 8> v_frag[8];
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_load(v_frag[d], sh_V + (kv * 8) * STRIDE_V + (pass * 64 + d * 8), STRIDE_V);
                }

                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    #pragma unroll
                    for (uint d = 0; d < 8; d++) {
                        simdgroup_multiply_accumulate(o_frag[r][d], p_frag[r], v_frag[d], o_frag[r][d]);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_store(o_frag[r][d], sh_O_tile + (sg_row_start + r * 8) * STRIDE_O_TILE + (pass * 64 + d * 8), STRIDE_O_TILE);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_q_tok = q_start + sg_row_start + r;
        if (global_q_tok < M && global_q_tok >= k_chunk_start) {
            bool is_done = (global_q_tok < (k_chunk_start + k_chunk_len)) || (is_last_chunk != 0);
            if (is_done) {
                half inv_sum = (l_prev[r] > 0.0f) ? (half)(1.0f / l_prev[r]) : 0.0h;
                device half* out_ptr = O + (h * M + global_q_tok) * D + simd_lane_id * ELEMS_PER_THREAD;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    out_ptr[e] = o_acc[r][e] * inv_sum;
                }
            } else {
                if (simd_lane_id == 0) {
                    m_state[h * M + global_q_tok] = m_prev[r];
                    l_state[h * M + global_q_tok] = l_prev[r];
                }
                device half* state_ptr = O_state + (h * M + global_q_tok) * D + simd_lane_id * ELEMS_PER_THREAD;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    state_ptr[e] = o_acc[r][e];
                }
            }
        }
    }
}

template<uint D>
inline void streaming_flash_attn_chunk_q8_0_impl(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8_chunk [[buffer(1)]],
    device const block_q8_0*   V_q8_chunk [[buffer(2)]],
    device half*               O [[buffer(3)]],
    device float*              m_state [[buffer(4)]],
    device float*              l_state [[buffer(5)]],
    device half*               O_state [[buffer(6)]],
    constant uint&             M [[buffer(7)]],
    constant uint&             C_slot [[buffer(8)]],
    constant uint&             k_chunk_start [[buffer(9)]],
    constant uint&             k_chunk_len [[buffer(10)]],
    constant uint&             is_first_chunk [[buffer(11)]],
    constant uint&             is_last_chunk [[buffer(12)]],
    constant float&            scale [[buffer(13)]],
    threadgroup half*          smem_raw [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BQ = 64;
    constexpr uint BK = 64;
    constexpr uint D_FRAGS = D / 8;     // 8 for D=64, 16 for D=128
    constexpr uint NUM_PASSES = D_FRAGS / 8;
    constexpr uint BLOCKS_PER_TOKEN = D / 32; // 2 for D=64, 4 for D=128
    constexpr uint TOTAL_KV_BLOCKS = 64 * BLOCKS_PER_TOKEN; // 128 for D=64, 256 for D=128
    constexpr uint BLOCKS_PER_THREAD = TOTAL_KV_BLOCKS / 128; // 1 for D=64, 2 for D=128

    constexpr uint STRIDE_Q = D;
    constexpr uint STRIDE_K_T = 64;
    constexpr uint STRIDE_V = D;
    constexpr uint STRIDE_S_P = 64;
    constexpr uint STRIDE_O_TILE = D;

    uint b_q = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    uint q_start = b_q * BQ;

    if (q_start >= M) return;

    if (q_start + BQ <= k_chunk_start) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    constexpr uint Q_SIZE = 64 * D;
    constexpr uint K_T_SIZE = D * 64;
    constexpr uint V_SIZE = 64 * D;

    threadgroup half* sh_Q      = smem_raw;
    threadgroup half* sh_K_T    = smem_raw + Q_SIZE;
    threadgroup half* sh_V      = smem_raw + Q_SIZE + K_T_SIZE;
    threadgroup half* sh_S_P    = smem_raw + Q_SIZE + K_T_SIZE + V_SIZE;
    threadgroup half* sh_O_tile = sh_K_T;

    uint sg_row_start = simd_group_id * 16;

    float m_prev[16];
    float l_prev[16];
    constexpr uint ELEMS_PER_THREAD = D / 32;
    half o_acc[16][ELEMS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_q_tok = q_start + sg_row_start + r;
        if (is_first_chunk != 0) {
            m_prev[r] = -1e30f;
            l_prev[r] = 0.0f;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = 0.0h;
            }
        } else {
            if (global_q_tok < M && global_q_tok >= k_chunk_start) {
                m_prev[r] = m_state[h * M + global_q_tok];
                l_prev[r] = l_state[h * M + global_q_tok];
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    uint col = simd_lane_id * ELEMS_PER_THREAD + e;
                    o_acc[r][e] = O_state[(h * M + global_q_tok) * D + col];
                }
            } else {
                m_prev[r] = -1e30f;
                l_prev[r] = 0.0f;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    o_acc[r][e] = 0.0h;
                }
            }
        }
    }

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

    uint num_k_tiles = (k_chunk_len + BK - 1) / BK;
    uint max_causal_k_tile = (min((b_q + 1) * BQ, M) - 1 >= k_chunk_start) ? ((min((b_q + 1) * BQ, M) - 1 - k_chunk_start) / BK) : 0;
    uint loop_k_tiles = min(max_causal_k_tile + 1, num_k_tiles);

    for (uint b_k = 0; b_k < loop_k_tiles; b_k++) {
        uint k_tile_local = b_k * BK;
        uint k_tile_global = k_chunk_start + k_tile_local;

        #pragma unroll
        for (uint b = 0; b < BLOCKS_PER_THREAD; b++) {
            uint blk_local_idx = linear_tid * BLOCKS_PER_THREAD + b;
            uint k_row = blk_local_idx / BLOCKS_PER_TOKEN;
            uint sub_blk = blk_local_idx % BLOCKS_PER_TOKEN;
            uint k_local_tok = k_tile_local + k_row;
            uint global_k_tok = k_tile_global + k_row;

            if (global_k_tok < M && k_local_tok < k_chunk_len) {
                uint slot_blk_idx = (h * C_slot + k_local_tok) * BLOCKS_PER_TOKEN + sub_blk;
                block_q8_0 k_blk = K_q8_chunk[slot_blk_idx];
                block_q8_0 v_blk = V_q8_chunk[slot_blk_idx];

                half kd = k_blk.d;
                half vd = v_blk.d;

                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    half k_val = (half)k_blk.qs[i] * kd;
                    half v_val = (half)v_blk.qs[i] * vd;

                    sh_K_T[(sub_blk * 32 + i) * 64 + k_row] = k_val;
                    sh_V[k_row * D + sub_blk * 32 + i] = v_val;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    sh_K_T[(sub_blk * 32 + i) * 64 + k_row] = 0.0h;
                    sh_V[k_row * D + sub_blk * 32 + i] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                simdgroup_store(s_frag[r][c], sh_S_P + (sg_row_start + r * 8) * STRIDE_S_P + (c * 8), STRIDE_S_P);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int r = 0; r < 16; r++) {
            uint global_q_tok = q_start + sg_row_start + r;
            uint c_a = simd_lane_id;
            uint c_b = simd_lane_id + 32;
            uint global_k_tok_a = k_chunk_start + k_tile_local + c_a;
            uint global_k_tok_b = k_chunk_start + k_tile_local + c_b;

            float s_a = -1e30f;
            if (global_q_tok < M && global_k_tok_a <= global_q_tok && global_k_tok_a < M && (k_tile_local + c_a) < k_chunk_len) {
                s_a = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] * scale;
            }

            float s_b = -1e30f;
            if (global_q_tok < M && global_k_tok_b <= global_q_tok && global_k_tok_b < M && (k_tile_local + c_b) < k_chunk_len) {
                s_b = (float)sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] * scale;
            }

            float block_max = max(s_a, s_b);
            block_max = max(block_max, simd_shuffle_down(block_max, 16));
            block_max = max(block_max, simd_shuffle_down(block_max, 8));
            block_max = max(block_max, simd_shuffle_down(block_max, 4));
            block_max = max(block_max, simd_shuffle_down(block_max, 2));
            block_max = max(block_max, simd_shuffle_down(block_max, 1));
            block_max = simd_broadcast(block_max, 0);

            float new_max = max(m_prev[r], block_max);
            float alpha = (m_prev[r] > -1e20f) ? metal::fast::exp(m_prev[r] - new_max) : 0.0f;
            float beta = (block_max > -1e20f) ? metal::fast::exp(block_max - new_max) : 0.0f;

            float p_a = (s_a > -1e20f) ? (metal::fast::exp(s_a - block_max) * beta) : 0.0f;
            float p_b = (s_b > -1e20f) ? (metal::fast::exp(s_b - block_max) * beta) : 0.0f;

            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_a] = (half)p_a;
            sh_S_P[(sg_row_start + r) * STRIDE_S_P + c_b] = (half)p_b;

            float block_sum = p_a + p_b;
            block_sum += simd_shuffle_down(block_sum, 16);
            block_sum += simd_shuffle_down(block_sum, 8);
            block_sum += simd_shuffle_down(block_sum, 4);
            block_sum += simd_shuffle_down(block_sum, 2);
            block_sum += simd_shuffle_down(block_sum, 1);
            block_sum = simd_broadcast(block_sum, 0);

            float new_sum = l_prev[r] * alpha + block_sum;
            m_prev[r] = new_max;
            l_prev[r] = new_sum;

            half alpha_h = (half)alpha;
            #pragma unroll
            for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[r][e] = o_acc[r][e] * alpha_h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint pass = 0; pass < NUM_PASSES; pass++) {
            simdgroup_matrix<half, 8, 8> o_frag[2][8];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    o_frag[r][d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                }
            }

            #pragma unroll
            for (uint kv = 0; kv < 8; kv++) {
                simdgroup_matrix<half, 8, 8> p_frag[2];
                simdgroup_load(p_frag[0], sh_S_P + (sg_row_start + 0) * STRIDE_S_P + (kv * 8), STRIDE_S_P);
                simdgroup_load(p_frag[1], sh_S_P + (sg_row_start + 8) * STRIDE_S_P + (kv * 8), STRIDE_S_P);

                simdgroup_matrix<half, 8, 8> v_frag[8];
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_load(v_frag[d], sh_V + (kv * 8) * STRIDE_V + (pass * 64 + d * 8), STRIDE_V);
                }

                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    #pragma unroll
                    for (uint d = 0; d < 8; d++) {
                        simdgroup_multiply_accumulate(o_frag[r][d], p_frag[r], v_frag[d], o_frag[r][d]);
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (uint d = 0; d < 8; d++) {
                    simdgroup_store(o_frag[r][d], sh_O_tile + (sg_row_start + r * 8) * STRIDE_O_TILE + (pass * 64 + d * 8), STRIDE_O_TILE);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        uint global_q_tok = q_start + sg_row_start + r;
        if (global_q_tok < M && global_q_tok >= k_chunk_start) {
            bool is_done = (global_q_tok < (k_chunk_start + k_chunk_len)) || (is_last_chunk != 0);
            if (is_done) {
                half inv_sum = (l_prev[r] > 0.0f) ? (half)(1.0f / l_prev[r]) : 0.0h;
                device half* out_ptr = O + (h * M + global_q_tok) * D + simd_lane_id * ELEMS_PER_THREAD;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    out_ptr[e] = o_acc[r][e] * inv_sum;
                }
            } else {
                if (simd_lane_id == 0) {
                    m_state[h * M + global_q_tok] = m_prev[r];
                    l_state[h * M + global_q_tok] = l_prev[r];
                }
                device half* state_ptr = O_state + (h * M + global_q_tok) * D + simd_lane_id * ELEMS_PER_THREAD;
                #pragma unroll
                for (uint e = 0; e < ELEMS_PER_THREAD; e++) {
                    state_ptr[e] = o_acc[r][e];
                }
            }
        }
    }
}

// Chunked Streaming Kernel Entry Points
kernel void streaming_flash_attn_chunk_fp16_d64(
    device const half* Q [[buffer(0)]],
    device const half* K_chunk [[buffer(1)]],
    device const half* V_chunk [[buffer(2)]],
    device half*       O [[buffer(3)]],
    device float*      m_state [[buffer(4)]],
    device float*      l_state [[buffer(5)]],
    device half*       O_state [[buffer(6)]],
    constant uint&     M [[buffer(7)]],
    constant uint&     C_slot [[buffer(8)]],
    constant uint&     k_chunk_start [[buffer(9)]],
    constant uint&     k_chunk_len [[buffer(10)]],
    constant uint&     is_first_chunk [[buffer(11)]],
    constant uint&     is_last_chunk [[buffer(12)]],
    constant float&    scale [[buffer(13)]],
    threadgroup half*  smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    streaming_flash_attn_chunk_fp16_impl<64>(
        Q, K_chunk, V_chunk, O, m_state, l_state, O_state,
        M, C_slot, k_chunk_start, k_chunk_len, is_first_chunk, is_last_chunk, scale,
        smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void streaming_flash_attn_chunk_fp16_d128(
    device const half* Q [[buffer(0)]],
    device const half* K_chunk [[buffer(1)]],
    device const half* V_chunk [[buffer(2)]],
    device half*       O [[buffer(3)]],
    device float*      m_state [[buffer(4)]],
    device float*      l_state [[buffer(5)]],
    device half*       O_state [[buffer(6)]],
    constant uint&     M [[buffer(7)]],
    constant uint&     C_slot [[buffer(8)]],
    constant uint&     k_chunk_start [[buffer(9)]],
    constant uint&     k_chunk_len [[buffer(10)]],
    constant uint&     is_first_chunk [[buffer(11)]],
    constant uint&     is_last_chunk [[buffer(12)]],
    constant float&    scale [[buffer(13)]],
    threadgroup half*  smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    streaming_flash_attn_chunk_fp16_impl<128>(
        Q, K_chunk, V_chunk, O, m_state, l_state, O_state,
        M, C_slot, k_chunk_start, k_chunk_len, is_first_chunk, is_last_chunk, scale,
        smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void streaming_flash_attn_chunk_q8_0_d64(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8_chunk [[buffer(1)]],
    device const block_q8_0*   V_q8_chunk [[buffer(2)]],
    device half*               O [[buffer(3)]],
    device float*              m_state [[buffer(4)]],
    device float*              l_state [[buffer(5)]],
    device half*               O_state [[buffer(6)]],
    constant uint&             M [[buffer(7)]],
    constant uint&             C_slot [[buffer(8)]],
    constant uint&             k_chunk_start [[buffer(9)]],
    constant uint&             k_chunk_len [[buffer(10)]],
    constant uint&             is_first_chunk [[buffer(11)]],
    constant uint&             is_last_chunk [[buffer(12)]],
    constant float&            scale [[buffer(13)]],
    threadgroup half*          smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    streaming_flash_attn_chunk_q8_0_impl<64>(
        Q, K_q8_chunk, V_q8_chunk, O, m_state, l_state, O_state,
        M, C_slot, k_chunk_start, k_chunk_len, is_first_chunk, is_last_chunk, scale,
        smem, tg_pos, simd_lane_id, simd_group_id);
}

kernel void streaming_flash_attn_chunk_q8_0_d128(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8_chunk [[buffer(1)]],
    device const block_q8_0*   V_q8_chunk [[buffer(2)]],
    device half*               O [[buffer(3)]],
    device float*              m_state [[buffer(4)]],
    device float*              l_state [[buffer(5)]],
    device half*               O_state [[buffer(6)]],
    constant uint&             M [[buffer(7)]],
    constant uint&             C_slot [[buffer(8)]],
    constant uint&             k_chunk_start [[buffer(9)]],
    constant uint&             k_chunk_len [[buffer(10)]],
    constant uint&             is_first_chunk [[buffer(11)]],
    constant uint&             is_last_chunk [[buffer(12)]],
    constant float&            scale [[buffer(13)]],
    threadgroup half*          smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    streaming_flash_attn_chunk_q8_0_impl<128>(
        Q, K_q8_chunk, V_q8_chunk, O, m_state, l_state, O_state,
        M, C_slot, k_chunk_start, k_chunk_len, is_first_chunk, is_last_chunk, scale,
        smem, tg_pos, simd_lane_id, simd_group_id);
}
