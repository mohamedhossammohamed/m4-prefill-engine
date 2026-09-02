#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA TYPES & HELPERS
// ============================================================================
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// Fast vector exp for float4
inline float4 fast_exp(float4 x) {
    return exp(x);
}

// Fast SiLU for float
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

// Fast SiLU for float4
inline float4 silu_f32x4(float4 x) {
    return x / (float4(1.0f) + exp(-x));
}

// ============================================================================
// 1. HARDWARE ROOFLINE & MEMORY PROBES
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
// 2. VECTORIZED ELEMENTWISE KERNELS (RESIDUAL & SWIGLU & QUANTIZATION)
// ============================================================================

// High-Throughput Residual Addition: Out = In1 + In2 (128-bit vector transactions)
kernel void vector_add_residual(
    device const half4* in1 [[buffer(0)]],
    device const half4* in2 [[buffer(1)]],
    device half4*       out [[buffer(2)]],
    constant uint&      num_half4 [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < num_half4) {
        out[id] = in1[id] + in2[id];
    }
}

// High-Throughput Standalone SwiGLU Activation: S = SiLU(Gate) * Up
kernel void swiglu_activation(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base_idx = id * 4;
    if (base_idx < num_elements) {
        device const half4* g_ptr = reinterpret_cast<device const half4*>(gate + base_idx);
        device const half4* u_ptr = reinterpret_cast<device const half4*>(up + base_idx);
        device half4* o_ptr = reinterpret_cast<device half4*>(out + base_idx);

        half4 g_h = *g_ptr;
        half4 u_h = *u_ptr;

        float4 g_f = float4(g_h);
        float4 u_f = float4(u_h);

        float4 silu_g = g_f / (float4(1.0f) + exp(-g_f));
        float4 res = silu_g * u_f;

        *o_ptr = half4(res);
    }
}

// Dynamic FP16 to Q8_0 KV Cache Quantization
kernel void quantize_kv_to_q8_0(
    device const half*       src [[buffer(0)]],
    device block_q8_0*       dst [[buffer(1)]],
    constant uint&           num_blocks [[buffer(2)]],
    uint blk_id [[thread_position_in_grid]])
{
    if (blk_id >= num_blocks) return;

    device const half* s = src + blk_id * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float val = fabs((float)s[i]);
        if (val > amax) amax = val;
    }

    float d = amax / 127.0f;
    dst[blk_id].d = (half)d;
    float id_scale = (d > 0.0f) ? (1.0f / d) : 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = (int)round((float)s[i] * id_scale);
        q = clamp(q, -128, 127);
        dst[blk_id].qs[i] = (int8_t)q;
    }
}

// Layout Transpose: [M, H * D] -> [H, M, D] (for D=128: 32 half4s per token)
kernel void transpose_m_hd_to_h_m_d(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint&     M   [[buffer(2)]],
    constant uint&     H   [[buffer(3)]],
    constant uint&     D   [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint tok = pos.x;
    uint h = pos.y;
    if (tok < M && h < H) {
        device const half* s = src + tok * (H * D) + h * D;
        device half* d = dst + (h * M + tok) * D;
        device const half4* s4 = reinterpret_cast<device const half4*>(s);
        device half4* d4 = reinterpret_cast<device half4*>(d);
        uint num_vec = D / 4;
        #pragma unroll
        for (uint i = 0; i < num_vec; i++) {
            d4[i] = s4[i];
        }
    }
}

// Layout Transpose: [H, M, D] -> [M, H * D] (for D=128: 32 half4s per token)
kernel void transpose_h_m_d_to_m_hd(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint&     M   [[buffer(2)]],
    constant uint&     H   [[buffer(3)]],
    constant uint&     D   [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint tok = pos.x;
    uint h = pos.y;
    if (tok < M && h < H) {
        device const half* s = src + (h * M + tok) * D;
        device half* d = dst + tok * (H * D) + h * D;
        device const half4* s4 = reinterpret_cast<device const half4*>(s);
        device half4* d4 = reinterpret_cast<device half4*>(d);
        uint num_vec = D / 4;
        #pragma unroll
        for (uint i = 0; i < num_vec; i++) {
            d4[i] = s4[i];
        }
    }
}

// ============================================================================
// 3. PIPELINED QUEUE-SATURATED DOUBLE-BUFFERED Q4_0 GEMM (32x32)
// ============================================================================
// Handles K=4096, N=4096 (O-proj), N=14336 (MLP Up), K=14336, N=4096 (MLP Down)
kernel void pipe_gemm_q4_0_32x32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    constant uint& weights_k_major [[buffer(6)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][32][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
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

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[weights_k_major ? col_idx : col_idx * num_k_blocks];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
            q_next = B[weights_k_major ? next_kb * N + col_idx
                                       : col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s0 = p0 + p1;
                half4 s1 = p2 + p3;
                half4 s = s0 + s1;

                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
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
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}

// ============================================================================
// 4. DIRECT HEAD-LAYOUT QKV GEMM PROJECTION (Fused Q4_0 Projection -> [H, M, D])
// ============================================================================
// Directly outputs Q, K, V in [H, M, D] layout for H=32, D=128 (Attn Dim = 4096, K = 4096)
kernel void pipe_qkv_head_gemm_q4_0(
    device const half*         A [[buffer(0)]], // [M, K]
    device const block_q4_0*   B [[buffer(1)]], // [H*D, K/32]
    device half*               C [[buffer(2)]], // [H, M, D]
    constant uint&             M [[buffer(3)]],
    constant uint&             H [[buffer(4)]],
    constant uint&             D [[buffer(5)]],
    constant uint&             K [[buffer(6)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][32][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint N = H * D;
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

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s0 = p0 + p1;
                half4 s1 = p2 + p3;
                half4 s = s0 + s1;

                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            q_curr = q_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        uint h = col_idx / D;
        uint d_offset = col_idx % D;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_m = tg_row_start + r;
            if (global_m < M) {
                // Direct scatter to layout [H, M, D]
                uint out_idx = (h * M + global_m) * D + d_offset;
                C[out_idx] = (half)acc[r];
            }
        }
    }
}

// ============================================================================
// 5. FUSED GATE + UP Q4_0 GEMM WITH IN-KERNEL SWIGLU EPILOGUE
// ============================================================================
// Computes SiLU(A @ W_gate) * (A @ W_up) directly into Out [M, N_mlp]
// Eliminates 2x [M, 14336] intermediate DRAM write & read roundtrips!
kernel void fused_gate_up_swiglu_q4_0(
    device const half*         A      [[buffer(0)]], // [M, K]
    device const block_q4_0*   B_gate [[buffer(1)]], // [N_mlp, K/32]
    device const block_q4_0*   B_up   [[buffer(2)]], // [N_mlp, K/32]
    device half*               Out    [[buffer(3)]], // [M, N_mlp]
    constant uint&             M      [[buffer(4)]],
    constant uint&             N_mlp  [[buffer(5)]],
    constant uint&             K      [[buffer(6)]],
    threadgroup half*          shmem  [[threadgroup(0)]], // [2][64][32] + staged Gate/Up Q4 blocks
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N_mlp);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[64][32] = (threadgroup half (*)[64][32])shmem;
    threadgroup block_q4_0 (*sh_B_gate)[32] =
        (threadgroup block_q4_0 (*)[32])(&sh_A[2][0][0]);
    threadgroup block_q4_0 (*sh_B_up)[32] = sh_B_gate + 2;

    float acc_g[32];
    float acc_u[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc_g[r] = 0.0f;
        acc_u[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint tile_r = simd_group_id * 32 + r;
            uint global_r = tg_row_start + tile_r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][tile_r][c]) = val;
        }
    };

    load_A(0, 0);
    if (simd_group_id == 0 && valid_col) {
            sh_B_gate[0][simd_lane_id] = B_gate[col_idx];
            sh_B_up[0][simd_lane_id] = B_up[col_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 qg_curr, qu_curr;

        if (valid_col) {
            qg_curr = sh_B_gate[cur_buf][simd_lane_id];
            qu_curr = sh_B_up[cur_buf][simd_lane_id];
        }

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (simd_group_id == 0 && valid_col) {
                sh_B_gate[nxt_buf][simd_lane_id] =
                B_gate[next_kb * N_mlp + col_idx];
            sh_B_up[nxt_buf][simd_lane_id] =
                B_up[next_kb * N_mlp + col_idx];
            }
        }

        if (valid_col) {
            // Unpack Gate weights
            half dg = qg_curr.d;
            half4 hdg = half4(dg);
            half4 h_off_g = half4(-8.0h * dg);
            uint gw0 = read_u32_unaligned(qg_curr.qs + 0);
            uint gw1 = read_u32_unaligned(qg_curr.qs + 4);
            uint gw2 = read_u32_unaligned(qg_curr.qs + 8);
            uint gw3 = read_u32_unaligned(qg_curr.qs + 12);

            half4 g_low[4], g_high[4];
            g_low[0]  = fma(half4(as_type<uchar4>(gw0 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[0] = fma(half4(as_type<uchar4>((gw0 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[1]  = fma(half4(as_type<uchar4>(gw1 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[1] = fma(half4(as_type<uchar4>((gw1 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[2]  = fma(half4(as_type<uchar4>(gw2 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[2] = fma(half4(as_type<uchar4>((gw2 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[3]  = fma(half4(as_type<uchar4>(gw3 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[3] = fma(half4(as_type<uchar4>((gw3 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);

            // Unpack Up weights
            half du = qu_curr.d;
            half4 hdu = half4(du);
            half4 h_off_u = half4(-8.0h * du);
            uint uw0 = read_u32_unaligned(qu_curr.qs + 0);
            uint uw1 = read_u32_unaligned(qu_curr.qs + 4);
            uint uw2 = read_u32_unaligned(qu_curr.qs + 8);
            uint uw3 = read_u32_unaligned(qu_curr.qs + 12);

            half4 u_low[4], u_high[4];
            u_low[0]  = fma(half4(as_type<uchar4>(uw0 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[0] = fma(half4(as_type<uchar4>((uw0 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[1]  = fma(half4(as_type<uchar4>(uw1 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[1] = fma(half4(as_type<uchar4>((uw1 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[2]  = fma(half4(as_type<uchar4>(uw2 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[2] = fma(half4(as_type<uchar4>((uw2 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[3]  = fma(half4(as_type<uchar4>(uw3 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[3] = fma(half4(as_type<uchar4>((uw3 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                // Full tiles stay on the original unrolled path. In the final
                // partial tile, skip dot products for rows that will not be stored.
                uint simd_row_start = tg_row_start + simd_group_id * 32;
                if (simd_row_start + 32 > M && simd_row_start + (uint)r >= M) {
                    continue;
                }

                uint tile_r = simd_group_id * 32 + r;
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][tile_r][28]);

                // Gate dot product
                half4 pg0 = a0 * g_low[0] + a1 * g_low[1];
                half4 pg1 = a2 * g_low[2] + a3 * g_low[3];
                half4 pg2 = a4 * g_high[0] + a5 * g_high[1];
                half4 pg3 = a6 * g_high[2] + a7 * g_high[3];
                half4 sg = (pg0 + pg1) + (pg2 + pg3);
                acc_g[r] += (float)(sg[0] + sg[1] + sg[2] + sg[3]);

                // Up dot product
                half4 pu0 = a0 * u_low[0] + a1 * u_low[1];
                half4 pu1 = a2 * u_low[2] + a3 * u_low[3];
                half4 pu2 = a4 * u_high[0] + a5 * u_high[1];
                half4 pu3 = a6 * u_high[2] + a7 * u_high[3];
                half4 su = (pu0 + pu1) + (pu2 + pu3);
                acc_u[r] += (float)(su[0] + su[1] + su[2] + su[3]);
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + simd_group_id * 32 + r;
            if (global_r < M) {
                // SwiGLU Activation: S = SiLU(Gate) * Up
                float g = acc_g[r];
                float u = acc_u[r];
                float silu_g = g / (1.0f + exp(-g));
                float swiglu = silu_g * u;
                Out[global_r * N_mlp + col_idx] = (half)swiglu;
            }
        }
    }
}

// ============================================================================
// 6. FUSED FLASHATTENTION ENGINE WITH D=128 CAUSAL MASKING (FP16 & Q8_0)
// ============================================================================

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_fp16_d128(
    device const half* K,
    device const half* V,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    // D=128 has 32 half4 vectors per token
    constexpr uint total_half4 = (BC * 128) / 4;
    #pragma unroll
    for (uint vec_idx = tid; vec_idx < total_half4; vec_idx += TG_SIZE) {
        uint row = vec_idx / 32;
        uint col_vec = vec_idx % 32;
        uint global_tok = c_start + row;
        
        threadgroup half4* k_dst = (threadgroup half4*)(smem_K + row * 128);
        threadgroup half4* v_dst = (threadgroup half4*)(smem_V + row * 128);
        
        if (global_tok < M) {
            device const half4* k_src = (device const half4*)(K + (h * M + global_tok) * 128);
            device const half4* v_src = (device const half4*)(V + (h * M + global_tok) * 128);
            k_dst[col_vec] = k_src[col_vec];
            v_dst[col_vec] = v_src[col_vec];
        } else {
            k_dst[col_vec] = half4(0.0h);
            v_dst[col_vec] = half4(0.0h);
        }
    }
}

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_q8_0_d128(
    device const block_q8_0* K_q8,
    device const block_q8_0* V_q8,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    // D=128 has 4 block_q8_0 (32 elements each) per token
    constexpr uint total_blocks = BC * 4;
    #pragma unroll
    for (uint blk_idx = tid; blk_idx < total_blocks; blk_idx += TG_SIZE) {
        uint row = blk_idx / 4;
        uint sub_blk = blk_idx % 4;
        uint global_tok = c_start + row;
        
        threadgroup half* k_dst = smem_K + row * 128 + sub_blk * 32;
        threadgroup half* v_dst = smem_V + row * 128 + sub_blk * 32;
        
        if (global_tok < M) {
            uint blk_offset = (h * M + global_tok) * 4 + sub_blk;
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
            threadgroup half4* k_dst4 = (threadgroup half4*)(smem_K + row * 128 + sub_blk * 32);
            threadgroup half4* v_dst4 = (threadgroup half4*)(smem_V + row * 128 + sub_blk * 32);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                k_dst4[i] = half4(0.0h);
                v_dst4[i] = half4(0.0h);
            }
        }
    }
}

// FlashAttention FP16 for D=128 (32x16 Tile - Optimal Latency & High Register Occupancy on M4)
kernel void flash_attn_fp16_causal_d128(
    device const half* Q     [[buffer(0)]], // [H, M, 128]
    device const half* K     [[buffer(1)]], // [H, M, 128]
    device const half* V     [[buffer(2)]], // [H, M, 128]
    device half*       O     [[buffer(3)]], // [M, H * 128] output directly in [M, H*D] layout!
    constant uint&     M     [[buffer(4)]],
    constant uint&     H     [[buffer(5)]],
    constant float&    scale [[buffer(6)]],
    threadgroup half*  shmem [[threadgroup(0)]], // [2][16*128 + 16*128] = 16KB
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    constexpr ushort BR = 64;
    constexpr ushort BC = 16;
    constexpr ushort TG_SIZE = 64;

    uint b_r = tg_pos.x; // Query tile index
    uint h   = tg_pos.y; // Head index
    
    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);
    
    // Register allocation for Q row (128 halfs = 32 half4s)
    half4 q_reg[32];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * 128);
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            q_reg[d] = half4(0.0h);
        }
    }
    
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[32];
    #pragma unroll
    for (int d = 0; d < 32; d++) {
        o_acc[d] = half4(0.0h);
    }
    
    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);
    
    threadgroup half (*smem_K)[BC * 128] = (threadgroup half (*)[BC * 128])shmem;
    threadgroup half (*smem_V)[BC * 128] = (threadgroup half (*)[BC * 128])(shmem + 2 * BC * 128);

    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile_fp16_d128<TG_SIZE, BC>(K, V, smem_K[0], smem_V[0], h, 0, M, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint b_c = 0; b_c < loop_tiles; b_c++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_b_c = b_c + 1;
        if (next_b_c < loop_tiles) {
            load_kv_tile_fp16_d128<TG_SIZE, BC>(K, V, smem_K[nxt_buf], smem_V[nxt_buf], h, next_b_c * BC, M, tid);
        }
        
        if (is_valid_row) {
            float s_tile[BC];
            float tile_local_max = -1e30f;
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                uint col_idx = b_c * BC + c;
                if (col_idx <= row_idx && col_idx < M) {
                    threadgroup const half4* k_ptr = (threadgroup const half4*)(smem_K[cur_buf] + c * 128);
                    half4 dot4 = half4(0.0h);
                    #pragma unroll
                    for (int d = 0; d < 32; d++) {
                        dot4 += q_reg[d] * k_ptr[d];
                    }
                    float dot = (float)(dot4[0] + dot4[1] + dot4[2] + dot4[3]) * scale;
                    s_tile[c] = dot;
                    if (dot > tile_local_max) tile_local_max = dot;
                } else {
                    s_tile[c] = -1e30f;
                }
            }
            
            float new_max = max(running_max, tile_local_max);
            float alpha = (running_max > -1e20f) ? exp(running_max - new_max) : 0.0f;
            running_max = new_max;
            running_sum = running_sum * alpha;
            
            #pragma unroll
            for (int d = 0; d < 32; d++) {
                o_acc[d] = o_acc[d] * (half)alpha;
            }
            
            float p_tile[BC];
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (s_tile[c] > -1e20f) {
                    float p = exp(s_tile[c] - running_max);
                    p_tile[c] = p;
                    running_sum += p;
                } else {
                    p_tile[c] = 0.0f;
                }
            }
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (p_tile[c] > 0.0f) {
                    half p_val = (half)p_tile[c];
                    threadgroup const half4* v_ptr = (threadgroup const half4*)(smem_V[cur_buf] + c * 128);
                    #pragma unroll
                    for (int d = 0; d < 32; d++) {
                        o_acc[d] = fma(v_ptr[d], half4(p_val), o_acc[d]);
                    }
                }
            }
        }
        
        if (next_b_c < loop_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cur_buf = nxt_buf;
        }
    }
    
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        // Directly output to layout [M, H * D]
        device half4* o_out = (device half4*)(O + row_idx * (H * 128) + h * 128);
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}

// FlashAttention Q8_0 for D=128 (32x16 Tile - Peak Throughput with Q8_0 KV Cache)
kernel void flash_attn_q8_0_causal_d128(
    device const half*       Q     [[buffer(0)]], // [H, M, 128]
    device const block_q8_0* K_q8  [[buffer(1)]], // [H, M, 4] blocks
    device const block_q8_0* V_q8  [[buffer(2)]], // [H, M, 4] blocks
    device half*             O     [[buffer(3)]], // [M, H * 128] output directly in [M, H*D] layout!
    constant uint&           M     [[buffer(4)]],
    constant uint&           H     [[buffer(5)]],
    constant float&          scale [[buffer(6)]],
    threadgroup half*        shmem [[threadgroup(0)]], // [2][16*128 + 16*128] = 16KB
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    constexpr ushort BR = 64;
    constexpr ushort BC = 16;
    constexpr ushort TG_SIZE = 64;

    uint b_r = tg_pos.x;
    uint h   = tg_pos.y;
    
    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);
    
    half4 q_reg[32];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * 128);
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            q_reg[d] = half4(0.0h);
        }
    }
    
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[32];
    #pragma unroll
    for (int d = 0; d < 32; d++) {
        o_acc[d] = half4(0.0h);
    }
    
    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);
    
    threadgroup half (*smem_K)[BC * 128] = (threadgroup half (*)[BC * 128])shmem;
    threadgroup half (*smem_V)[BC * 128] = (threadgroup half (*)[BC * 128])(shmem + 2 * BC * 128);

    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile_q8_0_d128<TG_SIZE, BC>(K_q8, V_q8, smem_K[0], smem_V[0], h, 0, M, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint b_c = 0; b_c < loop_tiles; b_c++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_b_c = b_c + 1;
        if (next_b_c < loop_tiles) {
            load_kv_tile_q8_0_d128<TG_SIZE, BC>(K_q8, V_q8, smem_K[nxt_buf], smem_V[nxt_buf], h, next_b_c * BC, M, tid);
        }
        
        if (is_valid_row) {
            float s_tile[BC];
            float tile_local_max = -1e30f;
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                uint col_idx = b_c * BC + c;
                if (col_idx <= row_idx && col_idx < M) {
                    threadgroup const half4* k_ptr = (threadgroup const half4*)(smem_K[cur_buf] + c * 128);
                    half4 dot4 = half4(0.0h);
                    #pragma unroll
                    for (int d = 0; d < 32; d++) {
                        dot4 += q_reg[d] * k_ptr[d];
                    }
                    float dot = (float)(dot4[0] + dot4[1] + dot4[2] + dot4[3]) * scale;
                    s_tile[c] = dot;
                    if (dot > tile_local_max) tile_local_max = dot;
                } else {
                    s_tile[c] = -1e30f;
                }
            }
            
            float new_max = max(running_max, tile_local_max);
            float alpha = (running_max > -1e20f) ? exp(running_max - new_max) : 0.0f;
            running_max = new_max;
            running_sum = running_sum * alpha;
            
            #pragma unroll
            for (int d = 0; d < 32; d++) {
                o_acc[d] = o_acc[d] * (half)alpha;
            }
            
            float p_tile[BC];
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (s_tile[c] > -1e20f) {
                    float p = exp(s_tile[c] - running_max);
                    p_tile[c] = p;
                    running_sum += p;
                } else {
                    p_tile[c] = 0.0f;
                }
            }
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (p_tile[c] > 0.0f) {
                    half p_val = (half)p_tile[c];
                    threadgroup const half4* v_ptr = (threadgroup const half4*)(smem_V[cur_buf] + c * 128);
                    #pragma unroll
                    for (int d = 0; d < 32; d++) {
                        o_acc[d] = fma(v_ptr[d], half4(p_val), o_acc[d]);
                    }
                }
            }
        }
        
        if (next_b_c < loop_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cur_buf = nxt_buf;
        }
    }
    
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        device half4* o_out = (device half4*)(O + row_idx * (H * 128) + h * 128);
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}

// ============================================================================
// 7. BASELINE KERNELS (LLAMA.CPP STYLE & NAIVE UN-FUSED)
// ============================================================================

// llama.cpp Style mul_mm_q4_0
kernel void llamacpp_style_mul_mm_q4_0(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          sh_A [[threadgroup(0)]], // [64 x 32] = 4KB
    threadgroup half*          sh_B [[threadgroup(1)]], // [32 x 32] = 2KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
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

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                half a_val = sh_A[(thread_r + r) * 32 + k];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    half b_val = sh_B[k * 32 + (thread_c + c)];
                    acc[r][c] += (float)a_val * (float)b_val;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

// Naive QK Causal Attention Baseline Stage 1
kernel void naive_attn_qk_causal(
    device const half* Q     [[buffer(0)]], // [H, M, D]
    device const half* K     [[buffer(1)]], // [H, M, D]
    device half*       S     [[buffer(2)]], // [H, M, M]
    constant uint&     M     [[buffer(3)]],
    constant uint&     D     [[buffer(4)]],
    constant float&    scale [[buffer(5)]],
    uint3 id [[thread_position_in_grid]])
{
    uint col_k = id.x; // Key index [0, M)
    uint row_q = id.y; // Query index [0, M)
    uint head  = id.z; // Head index [0, H)

    if (row_q >= M || col_k >= M) return;

    uint s_idx = head * (M * M) + row_q * M + col_k;

    if (col_k > row_q) {
        S[s_idx] = -INFINITY;
        return;
    }

    device const half* q_ptr = Q + (head * M + row_q) * D;
    device const half* k_ptr = K + (head * M + col_k) * D;

    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        acc += (float)q_ptr[d] * (float)k_ptr[d];
    }
    S[s_idx] = (half)(acc * scale);
}

// Naive Softmax Baseline Stage 2
kernel void naive_attn_softmax(
    device const half* S [[buffer(0)]], // [H, M, M]
    device half*       P [[buffer(1)]], // [H, M, M]
    constant uint&     M [[buffer(2)]],
    uint2 id [[thread_position_in_grid]])
{
    uint row_q = id.x; // Query index [0, M)
    uint head  = id.y; // Head index [0, H)

    if (row_q >= M) return;

    uint row_offset = head * (M * M) + row_q * M;
    device const half* s_row = S + row_offset;
    device half* p_row = P + row_offset;

    float max_val = -1e30f;
    for (uint k = 0; k <= row_q; k++) {
        float val = (float)s_row[k];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (uint k = 0; k <= row_q; k++) {
        float p = exp((float)s_row[k] - max_val);
        p_row[k] = (half)p;
        sum_exp += p;
    }

    for (uint k = row_q + 1; k < M; k++) {
        p_row[k] = 0.0h;
    }

    float inv_sum = 1.0f / sum_exp;
    for (uint k = 0; k <= row_q; k++) {
        p_row[k] = (half)((float)p_row[k] * inv_sum);
    }
}

// Naive PV Baseline Stage 3
kernel void naive_attn_pv(
    device const half* P [[buffer(0)]], // [H, M, M]
    device const half* V [[buffer(1)]], // [H, M, D]
    device half*       O [[buffer(2)]], // [H, M, D]
    constant uint&     M [[buffer(3)]],
    constant uint&     D [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    uint d_idx = id.x; // Head Dim [0, D)
    uint row_q = id.y; // Query index [0, M)
    uint head  = id.z; // Head index [0, H)

    if (d_idx >= D || row_q >= M) return;

    device const half* p_row = P + (head * (M * M) + row_q * M);
    float acc = 0.0f;

    for (uint k = 0; k <= row_q; k++) {
        float p = (float)p_row[k];
        if (p > 0.0f) {
            float v = (float)V[(head * M + k) * D + d_idx];
            acc += p * v;
        }
    }

    O[(head * M + row_q) * D + d_idx] = (half)acc;
}
