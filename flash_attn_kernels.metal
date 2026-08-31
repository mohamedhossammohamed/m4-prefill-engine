#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct block_q8_0 {
    half d;
    int8_t qs[32];
};

// ============================================================================
// HELPER: VECTOR LOADING AND DOUBLE BUFFERING COOPERATIVE LOADS
// ============================================================================

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_fp16(
    device const half* K,
    device const half* V,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    constexpr uint total_half4 = (BC * 64) / 4; // number of half4 vectors (512 for BC=32)
    #pragma unroll
    for (uint vec_idx = tid; vec_idx < total_half4; vec_idx += TG_SIZE) {
        uint row = vec_idx / 16;
        uint col_vec = vec_idx % 16;
        uint global_tok = c_start + row;
        
        threadgroup half4* k_dst = (threadgroup half4*)(smem_K + row * 64);
        threadgroup half4* v_dst = (threadgroup half4*)(smem_V + row * 64);
        
        if (global_tok < M) {
            device const half4* k_src = (device const half4*)(K + (h * M + global_tok) * 64);
            device const half4* v_src = (device const half4*)(V + (h * M + global_tok) * 64);
            k_dst[col_vec] = k_src[col_vec];
            v_dst[col_vec] = v_src[col_vec];
        } else {
            k_dst[col_vec] = half4(0.0h);
            v_dst[col_vec] = half4(0.0h);
        }
    }
}

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_q8_0(
    device const block_q8_0* K_q8,
    device const block_q8_0* V_q8,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    constexpr uint total_blocks = BC * 2; // 2 blocks of 32 per token for D=64 (64 for BC=32)
    #pragma unroll
    for (uint blk_idx = tid; blk_idx < total_blocks; blk_idx += TG_SIZE) {
        uint row = blk_idx / 2;
        uint sub_blk = blk_idx % 2;
        uint global_tok = c_start + row;
        
        threadgroup half* k_dst = smem_K + row * 64 + sub_blk * 32;
        threadgroup half* v_dst = smem_V + row * 64 + sub_blk * 32;
        
        if (global_tok < M) {
            uint blk_offset = (h * M + global_tok) * 2 + sub_blk;
            block_q8_0 k_blk = K_q8[blk_offset];
            block_q8_0 v_blk = V_q8[blk_offset];
            
            half kd = k_blk.d;
            half vd = v_blk.d;
            
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                k_dst[i] = (half)k_blk.qs[i] * kd;
                v_dst[i] = (half)v_blk.qs[i] * vd;
            }
        } else {
            threadgroup half4* k_dst4 = (threadgroup half4*)(smem_K + row * 64 + sub_blk * 32);
            threadgroup half4* v_dst4 = (threadgroup half4*)(smem_V + row * 64 + sub_blk * 32);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                k_dst4[i] = half4(0.0h);
                v_dst4[i] = half4(0.0h);
            }
        }
    }
}

// ============================================================================
// ULTRA-FAST VECTORIZED FLASHATTENTION IMPLEMENTATION (FP16 KV CACHE)
// ============================================================================
template<ushort TG_SIZE, ushort BR, ushort BC>
inline void flash_attn_fp16_impl(
    device const half* Q,
    device const half* K,
    device const half* V,
    device half*       O,
    constant uint&     M,
    constant float&    scale,
    threadgroup half (*smem_K)[BC * 64],
    threadgroup half (*smem_V)[BC * 64],
    uint2 tg_pos,
    uint tid)
{
    uint b_r = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    
    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);
    
    // Register allocation for Q row (64 halfs = 16 half4s)
    half4 q_reg[16];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = half4(0.0h);
        }
    }
    
    // Online Softmax State in Registers
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[16];
    #pragma unroll
    for (int d = 0; d < 16; d++) {
        o_acc[d] = half4(0.0h);
    }
    
    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);
    
    // Double Buffering: Prolog
    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile_fp16<TG_SIZE, BC>(K, V, smem_K[0], smem_V[0], h, 0, M, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint bc = 0; bc < loop_tiles; bc++) {
        uint next_buf = 1 - cur_buf;
        uint next_bc = bc + 1;
        
        // Asynchronously issue loads for next tile into double buffer
        if (next_bc < loop_tiles) {
            load_kv_tile_fp16<TG_SIZE, BC>(K, V, smem_K[next_buf], smem_V[next_buf], h, next_bc * BC, M, tid);
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
                    threadgroup const half4* k_row = (threadgroup const half4*)(smem_K[cur_buf] + j * 64);
                    
                    half4 acc0 = q_reg[0]*k_row[0] + q_reg[1]*k_row[1] + q_reg[2]*k_row[2] + q_reg[3]*k_row[3];
                    half4 acc1 = q_reg[4]*k_row[4] + q_reg[5]*k_row[5] + q_reg[6]*k_row[6] + q_reg[7]*k_row[7];
                    half4 acc2 = q_reg[8]*k_row[8] + q_reg[9]*k_row[9] + q_reg[10]*k_row[10] + q_reg[11]*k_row[11];
                    half4 acc3 = q_reg[12]*k_row[12] + q_reg[13]*k_row[13] + q_reg[14]*k_row[14] + q_reg[15]*k_row[15];
                    half4 tot = (acc0 + acc1) + (acc2 + acc3);
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
                for (int d = 0; d < 16; d++) {
                    o_acc[d] = o_acc[d] * alpha_h;
                }
                
                // 3. Ultra-fast FP16 FMA P * V accumulation
                #pragma unroll
                for (uint j = 0; j < BC; j++) {
                    half pj = P[j];
                    if (pj > 0.0h) {
                        threadgroup const half4* v_row = (threadgroup const half4*)(smem_V[cur_buf] + j * 64);
                        #pragma unroll
                        for (int d = 0; d < 16; d++) {
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
    
    // Epilog: write normalized output to global memory
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        device half4* o_out = (device half4*)(O + (h * M + row_idx) * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}

// ============================================================================
// ULTRA-FAST VECTORIZED FLASHATTENTION IMPLEMENTATION (Q8_0 KV CACHE)
// ============================================================================
template<ushort TG_SIZE, ushort BR, ushort BC>
inline void flash_attn_q8_0_impl(
    device const half*         Q,
    device const block_q8_0*   K_q8,
    device const block_q8_0*   V_q8,
    device half*               O,
    constant uint&             M,
    constant float&            scale,
    threadgroup half (*smem_K)[BC * 64],
    threadgroup half (*smem_V)[BC * 64],
    uint2 tg_pos,
    uint tid)
{
    uint b_r = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    
    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);
    
    // Register allocation for Q row (64 halfs = 16 half4s)
    half4 q_reg[16];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = half4(0.0h);
        }
    }
    
    // Online Softmax State in Registers
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[16];
    #pragma unroll
    for (int d = 0; d < 16; d++) {
        o_acc[d] = half4(0.0h);
    }
    
    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);
    
    // Double Buffering: Prolog
    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile_q8_0<TG_SIZE, BC>(K_q8, V_q8, smem_K[0], smem_V[0], h, 0, M, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint bc = 0; bc < loop_tiles; bc++) {
        uint next_buf = 1 - cur_buf;
        uint next_bc = bc + 1;
        
        // Asynchronously issue loads and on-the-fly SIMD dequantization for next tile
        if (next_bc < loop_tiles) {
            load_kv_tile_q8_0<TG_SIZE, BC>(K_q8, V_q8, smem_K[next_buf], smem_V[next_buf], h, next_bc * BC, M, tid);
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
                    threadgroup const half4* k_row = (threadgroup const half4*)(smem_K[cur_buf] + j * 64);
                    
                    half4 acc0 = q_reg[0]*k_row[0] + q_reg[1]*k_row[1] + q_reg[2]*k_row[2] + q_reg[3]*k_row[3];
                    half4 acc1 = q_reg[4]*k_row[4] + q_reg[5]*k_row[5] + q_reg[6]*k_row[6] + q_reg[7]*k_row[7];
                    half4 acc2 = q_reg[8]*k_row[8] + q_reg[9]*k_row[9] + q_reg[10]*k_row[10] + q_reg[11]*k_row[11];
                    half4 acc3 = q_reg[12]*k_row[12] + q_reg[13]*k_row[13] + q_reg[14]*k_row[14] + q_reg[15]*k_row[15];
                    half4 tot = (acc0 + acc1) + (acc2 + acc3);
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
                for (int d = 0; d < 16; d++) {
                    o_acc[d] = o_acc[d] * alpha_h;
                }
                
                // 3. Ultra-fast FP16 FMA P * V accumulation
                #pragma unroll
                for (uint j = 0; j < BC; j++) {
                    half pj = P[j];
                    if (pj > 0.0h) {
                        threadgroup const half4* v_row = (threadgroup const half4*)(smem_V[cur_buf] + j * 64);
                        #pragma unroll
                        for (int d = 0; d < 16; d++) {
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
    
    // Epilog: write normalized output to global memory
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        device half4* o_out = (device half4*)(O + (h * M + row_idx) * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}

// ============================================================================
// INSTANTIATIONS FOR PARAMETER SWEEP
// ============================================================================

// FP16 Kernels: (BR, BC) in [(16, 16), (32, 16), (32, 32), (64, 32)]
kernel void flash_attn_fp16_16x16(
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
    flash_attn_fp16_impl<32, 16, 16>(Q, K, V, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_fp16_32x16(
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
    flash_attn_fp16_impl<32, 32, 16>(Q, K, V, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_fp16_32x32(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][32 * 64];
    threadgroup half smem_V[2][32 * 64];
    flash_attn_fp16_impl<32, 32, 32>(Q, K, V, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_fp16_64x32(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][32 * 64];
    threadgroup half smem_V[2][32 * 64];
    flash_attn_fp16_impl<64, 64, 32>(Q, K, V, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

// Q8_0 Kernels: (BR, BC) in [(16, 16), (32, 16), (32, 32), (64, 32)]
kernel void flash_attn_q8_0_16x16(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][16 * 64];
    threadgroup half smem_V[2][16 * 64];
    flash_attn_q8_0_impl<32, 16, 16>(Q, K_q8, V_q8, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_q8_0_32x16(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][16 * 64];
    threadgroup half smem_V[2][16 * 64];
    flash_attn_q8_0_impl<32, 32, 16>(Q, K_q8, V_q8, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_q8_0_32x32(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][32 * 64];
    threadgroup half smem_V[2][32 * 64];
    flash_attn_q8_0_impl<32, 32, 32>(Q, K_q8, V_q8, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

kernel void flash_attn_q8_0_64x32(
    device const half*         Q [[buffer(0)]],
    device const block_q8_0*   K_q8 [[buffer(1)]],
    device const block_q8_0*   V_q8 [[buffer(2)]],
    device half*               O [[buffer(3)]],
    constant uint&             M [[buffer(4)]],
    constant float&            scale [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half smem_K[2][32 * 64];
    threadgroup half smem_V[2][32 * 64];
    flash_attn_q8_0_impl<64, 64, 32>(Q, K_q8, V_q8, O, M, scale, smem_K, smem_V, tg_pos, tid);
}

// ============================================================================
// NAIVE ATTENTION 3-STAGE BASELINE (Writes Full M x M Attention Matrix)
// ============================================================================

kernel void naive_attn_qk_causal(
    device const half* Q [[buffer(0)]],      // [H, M, D]
    device const half* K [[buffer(1)]],      // [H, M, D]
    device half*       S [[buffer(2)]],      // [H, M, M]
    constant uint&     M [[buffer(3)]],
    constant float&    scale [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]])
{
    uint col = pos.x; // key index j
    uint row = pos.y; // query index i
    uint h   = pos.z; // head index

    if (row >= M || col >= M) return;

    if (col > row) {
        S[(h * M + row) * M + col] = -INFINITY;
        return;
    }

    device const half4* q_row = (device const half4*)(Q + (h * M + row) * 64);
    device const half4* k_row = (device const half4*)(K + (h * M + col) * 64);

    half4 acc0 = q_row[0]*k_row[0] + q_row[1]*k_row[1] + q_row[2]*k_row[2] + q_row[3]*k_row[3];
    half4 acc1 = q_row[4]*k_row[4] + q_row[5]*k_row[5] + q_row[6]*k_row[6] + q_row[7]*k_row[7];
    half4 acc2 = q_row[8]*k_row[8] + q_row[9]*k_row[9] + q_row[10]*k_row[10] + q_row[11]*k_row[11];
    half4 acc3 = q_row[12]*k_row[12] + q_row[13]*k_row[13] + q_row[14]*k_row[14] + q_row[15]*k_row[15];
    half4 tot = (acc0 + acc1) + (acc2 + acc3);
    float dot = (float)(tot.x + tot.y + tot.z + tot.w);
    
    S[(h * M + row) * M + col] = (half)(dot * scale);
}

kernel void naive_attn_softmax(
    device const half* S [[buffer(0)]],      // [H, M, M]
    device half*       P [[buffer(1)]],      // [H, M, M]
    constant uint&     M [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.x; // query index i
    uint h   = pos.y; // head index

    if (row >= M) return;

    device const half* s_row = S + (h * M + row) * M;
    device half*       p_row = P + (h * M + row) * M;

    float max_val = -1e30f;
    for (uint j = 0; j <= row; j++) {
        float val = (float)s_row[j];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j <= row; j++) {
        float p = metal::fast::exp((float)s_row[j] - max_val);
        sum_exp += p;
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint j = 0; j < M; j++) {
        if (j <= row) {
            p_row[j] = (half)(metal::fast::exp((float)s_row[j] - max_val) * inv_sum);
        } else {
            p_row[j] = 0.0h;
        }
    }
}

kernel void naive_attn_pv(
    device const half* P [[buffer(0)]],      // [H, M, M]
    device const half* V [[buffer(1)]],      // [H, M, D]
    device half*       O [[buffer(2)]],      // [H, M, D]
    constant uint&     M [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]])
{
    uint d   = pos.x; // dim index [0..63]
    uint row = pos.y; // query index i
    uint h   = pos.z; // head index

    if (row >= M || d >= 64) return;

    device const half* p_row = P + (h * M + row) * M;
    device const half* v_base = V + h * M * 64 + d;

    float acc = 0.0f;
    for (uint j = 0; j <= row; j++) {
        acc += (float)p_row[j] * (float)v_base[j * 64];
    }
    O[(h * M + row) * 64 + d] = (half)acc;
}
