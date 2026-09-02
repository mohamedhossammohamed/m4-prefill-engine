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
// KERNEL 1: SCALAR BASELINE FLASHATTENTION (Vector ALU, BR=32, BC=16)
// ============================================================================
template<uint D>
inline void flash_attn_scalar_baseline_impl(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    threadgroup half*  smem_K_raw [[threadgroup(0)]],
    threadgroup half*  smem_V_raw [[threadgroup(1)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint BR = 32;
    constexpr uint BC = 16;
    constexpr uint D_VEC = D / 4;

    uint b_r = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index

    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);

    threadgroup half (*smem_K)[BC * D] = (threadgroup half (*)[BC * D])smem_K_raw;
    threadgroup half (*smem_V)[BC * D] = (threadgroup half (*)[BC * D])smem_V_raw;

    // Register allocation for Q row (D halfs = D_VEC half4s)
    half4 q_reg[D_VEC];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * D);
        #pragma unroll
        for (uint d = 0; d < D_VEC; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (uint d = 0; d < D_VEC; d++) {
            q_reg[d] = half4(0.0h);
        }
    }

    // Online Softmax State in Registers
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[D_VEC];
    #pragma unroll
    for (uint d = 0; d < D_VEC; d++) {
        o_acc[d] = half4(0.0h);
    }

    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);

    constexpr uint total_half4 = (BC * D) / 4;

    auto load_kv_tile = [&](uint cur_buf_idx, uint c_start) {
        #pragma unroll
        for (uint vec_idx = tid; vec_idx < total_half4; vec_idx += 32) {
            uint row = vec_idx / D_VEC;
            uint col_vec = vec_idx % D_VEC;
            uint global_tok = c_start + row;

            threadgroup half4* k_dst = (threadgroup half4*)(smem_K[cur_buf_idx] + row * D);
            threadgroup half4* v_dst = (threadgroup half4*)(smem_V[cur_buf_idx] + row * D);

            if (global_tok < M) {
                device const half4* k_src = (device const half4*)(K + (h * M + global_tok) * D);
                device const half4* v_src = (device const half4*)(V + (h * M + global_tok) * D);
                k_dst[col_vec] = k_src[col_vec];
                v_dst[col_vec] = v_src[col_vec];
            } else {
                k_dst[col_vec] = half4(0.0h);
                v_dst[col_vec] = half4(0.0h);
            }
        }
    };

    // Double Buffering: Prologue
    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile(0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bc = 0; bc < loop_tiles; bc++) {
        uint next_buf = 1 - cur_buf;
        uint next_bc = bc + 1;

        if (next_bc < loop_tiles) {
            load_kv_tile(next_buf, next_bc * BC);
        }

        if (is_valid_row) {
            uint c_start = bc * BC;
            float S[BC];
            float block_max = -1e30f;

            // 1. Vectorized Q * K^T with FP16 vector registers
            #pragma unroll
            for (uint j = 0; j < BC; j++) {
                uint global_k_tok = c_start + j;
                if (global_k_tok > row_idx || global_k_tok >= M) {
                    S[j] = -1e30f;
                } else {
                    threadgroup const half4* k_row = (threadgroup const half4*)(smem_K[cur_buf] + j * D);
                    half4 tot = half4(0.0h);
                    #pragma unroll
                    for (uint d = 0; d < D_VEC; d++) {
                        tot = fma(q_reg[d], k_row[d], tot);
                    }
                    float dot = (float)(tot.x + tot.y + tot.z + tot.w);
                    float s = dot * scale;
                    S[j] = s;
                    if (s > block_max) block_max = s;
                }
            }

            // 2. Online Softmax update
            if (block_max > -1e20f) {
                float new_max = max(running_max, block_max);
                float alpha = (running_max > -1e20f) ? metal::fast::exp(running_max - new_max) : 0.0f;
                float block_sum = 0.0f;
                half P[BC];

                #pragma unroll
                for (uint j = 0; j < BC; j++) {
                    if (S[j] > -1e20f) {
                        float p = metal::fast::exp(S[j] - new_max);
                        P[j] = (half)p;
                        block_sum += p;
                    } else {
                        P[j] = 0.0h;
                    }
                }

                running_sum = running_sum * alpha + block_sum;
                running_max = new_max;

                half alpha_h = (half)alpha;
                #pragma unroll
                for (uint d = 0; d < D_VEC; d++) {
                    o_acc[d] = o_acc[d] * alpha_h;
                }

                // 3. Ultra-fast FP16 FMA P * V accumulation
                #pragma unroll
                for (uint j = 0; j < BC; j++) {
                    half pj = P[j];
                    if (pj > 0.0h) {
                        threadgroup const half4* v_row = (threadgroup const half4*)(smem_V[cur_buf] + j * D);
                        #pragma unroll
                        for (uint d = 0; d < D_VEC; d++) {
                            o_acc[d] = fma(v_row[d], half4(pj), o_acc[d]);
                        }
                    }
                }
            }
        }

        if (next_bc < loop_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cur_buf = next_buf;
        }
    }

    // Epilogue: write normalized output to global memory
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        device half4* o_out = (device half4*)(O + (h * M + row_idx) * D);
        #pragma unroll
        for (uint d = 0; d < D_VEC; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}

kernel void flash_attn_scalar_baseline_d64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][16 * 64];
    threadgroup half smem_V[2][16 * 64];
    flash_attn_scalar_baseline_impl<64>(Q, K, V, O, M, scale, (threadgroup half*)smem_K, (threadgroup half*)smem_V, tg_pos, tid);
}

kernel void flash_attn_scalar_baseline_d128(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][16 * 128];
    threadgroup half smem_V[2][16 * 128];
    flash_attn_scalar_baseline_impl<128>(Q, K, V, O, M, scale, (threadgroup half*)smem_K, (threadgroup half*)smem_V, tg_pos, tid);
}

kernel void flash_attn_scalar_baseline(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][16 * 64];
    threadgroup half smem_V[2][16 * 64];
    flash_attn_scalar_baseline_impl<64>(Q, K, V, O, M, scale, (threadgroup half*)smem_K, (threadgroup half*)smem_V, tg_pos, tid);
}

// ============================================================================
// KERNEL 2: 2D BLOCK-MMA FLASHATTENTION (FP16 KV CACHE, BQ=64, BK=64)
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

        // D. Hardware Tensor-Core Matrix Multiply: O_tile = P_sg * V [16, D]
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

kernel void flash_attn_mma_64x64_fp16(
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

// ============================================================================
// KERNEL 3: 2D BLOCK-MMA FLASHATTENTION WITH DYNAMIC Q8_0 KV CACHE
// ============================================================================
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
    constexpr uint ELEMS_PER_THREAD = D / 32;
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

        // A. Dynamic On-The-Fly Q8_0 Dequantization into Threadgroup SRAM
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

                    // Transposed K into sh_K_T [D, 64]
                    sh_K_T[(sub_blk * 32 + i) * 64 + k_row] = k_val;
                    // Standard V into sh_V [64, D]
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

        // D. Hardware Tensor-Core Matrix Multiply: O_tile = P_sg * V [16, D]
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

kernel void flash_attn_mma_64x64_q8_0(
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
