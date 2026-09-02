#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA TYPES & HELPER FUNCTIONS
// ============================================================================
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// ============================================================================
// HARDWARE CEILING PROBES
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
// 1. KERNEL 1: swiglu_scalar_baseline
// Current baseline fused_gate_up_swiglu_q4_0
// Scalar Vector ALU with 64 float accumulators per thread (32 Gate + 32 Up)
// Single SIMDgroup (32 threads) computing [32, 32] tile
// ============================================================================
kernel void swiglu_scalar_baseline(
    device const half*         A      [[buffer(0)]], // [M, K]
    device const block_q4_0*   B_gate [[buffer(1)]], // [N_mlp, K/32]
    device const block_q4_0*   B_up   [[buffer(2)]], // [N_mlp, K/32]
    device half*               Out    [[buffer(3)]], // [M, N_mlp]
    constant uint&             M      [[buffer(4)]],
    constant uint&             N_mlp  [[buffer(5)]],
    constant uint&             K      [[buffer(6)]],
    threadgroup half*          shmem  [[threadgroup(0)]], // [2][32][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N_mlp);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32][32] = (threadgroup half (*)[32][32])shmem;

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
    block_q4_0 qg_curr, qu_curr;
    if (valid_col) {
        qg_curr = B_gate[col_idx * num_k_blocks + 0];
        qu_curr = B_up[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 qg_next, qu_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                qg_next = B_gate[col_idx * num_k_blocks + next_kb];
                qu_next = B_up[col_idx * num_k_blocks + next_kb];
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
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

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
            qg_curr = qg_next;
            qu_curr = qu_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
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
// 2. KERNEL 2: swiglu_split_mlx_style
// Multi-pass execution mimicking MLX architecture:
// Pass A: Gate GEMM (MMA Q4_0 -> Gate intermediate in DRAM)
// Pass B: Up GEMM   (MMA Q4_0 -> Up intermediate in DRAM)
// Pass C: Elementwise SiLU(Gate) * Up kernel
// ============================================================================

// GEMM MMA Q4_0 [64x64 Tile, 128 threads / 4 SIMDgroups] for Split Gate/Up GEMM
kernel void gemm_split_q4_0_pass(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // 16KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // sh_A: [64][32] = 2048 halfs (4KB)
    // sh_B: [32][64] = 2048 halfs (4KB)
    threadgroup half* sh_A = shmem;
    threadgroup half* sh_B = shmem + 2048;

    uint sg_r = simd_group_id / 2; // 0 or 1
    uint sg_c = simd_group_id % 2; // 0 or 1
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
        // 1. Cooperative coalesced loading of A: 64 rows x 32 cols
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

        // 2. Cooperative loading & on-the-fly dequantization of B (64 columns)
        if (linear_tid < 64) {
            uint b_col_idx = tg_col_start + linear_tid;
            if (b_col_idx < N) {
                block_q4_0 blk = B[b_col_idx * num_k_blocks + kb];
                half d = blk.d;
                half4 hd = half4(d);
                half4 h_off = half4(-8.0h * d);

                uint w0 = read_u32_unaligned(blk.qs + 0);
                uint w1 = read_u32_unaligned(blk.qs + 4);
                uint w2 = read_u32_unaligned(blk.qs + 8);
                uint w3 = read_u32_unaligned(blk.qs + 12);

                half4 vl[4], vh[4];
                vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
                vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
                vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
                vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
                vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sh_B[(i * 4 + 0) * 64 + linear_tid] = vl[i][0];
                    sh_B[(i * 4 + 1) * 64 + linear_tid] = vl[i][1];
                    sh_B[(i * 4 + 2) * 64 + linear_tid] = vl[i][2];
                    sh_B[(i * 4 + 3) * 64 + linear_tid] = vl[i][3];
                    sh_B[(16 + i * 4 + 0) * 64 + linear_tid] = vh[i][0];
                    sh_B[(16 + i * 4 + 1) * 64 + linear_tid] = vh[i][1];
                    sh_B[(16 + i * 4 + 2) * 64 + linear_tid] = vh[i][2];
                    sh_B[(16 + i * 4 + 3) * 64 + linear_tid] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B[k * 64 + linear_tid] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. MMA compute
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

    // 4. Staged store to global C via shared memory
    threadgroup float* sh_Out = (threadgroup float*)shmem;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            simdgroup_store(acc[r][c], &sh_Out[(sg_row_offset + r * 8) * 64 + (sg_col_offset + c * 8)], 64);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint elem_idx = linear_tid * 32 + i;
        uint r = elem_idx / 64;
        uint c = elem_idx % 64;
        uint global_r = tg_row_start + r;
        uint global_c = tg_col_start + c;
        if (global_r < M && global_c < N) {
            C[global_r * N + global_c] = (half)sh_Out[elem_idx];
        }
    }
}

// Pass C: Elementwise SiLU(Gate) * Up kernel
kernel void swiglu_split_elementwise_pass(
    device const half* Gate       [[buffer(0)]],
    device const half* Up         [[buffer(1)]],
    device half*       Out        [[buffer(2)]],
    constant uint&     total_elem [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < total_elem) {
        float g = (float)Gate[id];
        float u = (float)Up[id];
        float silu_g = g / (1.0f + exp(-g));
        Out[id] = (half)(silu_g * u);
    }
}

// Vectorized Pass C: 4-way vector ALU for maximum memory bandwidth saturation
kernel void swiglu_split_elementwise_vec4_pass(
    device const half4* Gate       [[buffer(0)]],
    device const half4* Up         [[buffer(1)]],
    device half4*       Out        [[buffer(2)]],
    constant uint&      total_vec4 [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < total_vec4) {
        half4 g = Gate[id];
        half4 u = Up[id];
        float4 gf = float4(g);
        float4 uf = float4(u);
        float4 silu_g = gf / (float4(1.0f) + exp(-gf));
        Out[id] = half4(silu_g * uf);
    }
}

// ============================================================================
// 3. KERNEL 3: swiglu_mma_dual_simd (Brick 3 Dual-SIMDgroup Cooperative Engine)
// - 128-thread threadgroup (4 SIMDgroups)
// - Input activation X is loaded once into threadgroup SRAM and shared across all 4 SIMDgroups
// - SIMDgroups 0 & 1 compute the Gate projection tile using simdgroup_matrix<half, 8, 8>
// - SIMDgroups 2 & 3 compute the Up projection tile using simdgroup_matrix<half, 8, 8>
// - Low register footprint (~26 regs/thread, 100% GPU core occupancy)
// - Epilogue: SIMDgroups write MMA fragments to threadgroup SRAM, compute SiLU(Gate) * Up
//   in on-chip SRAM, and write only the single activation tensor out to DRAM.
// ============================================================================
kernel void swiglu_mma_dual_simd(
    device const half*         A      [[buffer(0)]], // [M, K]
    device const block_q4_0*   B_gate [[buffer(1)]], // [N_mlp, K/32]
    device const block_q4_0*   B_up   [[buffer(2)]], // [N_mlp, K/32]
    device half*               Out    [[buffer(3)]], // [M, N_mlp]
    constant uint&             M      [[buffer(4)]],
    constant uint&             N_mlp  [[buffer(5)]],
    constant uint&             K      [[buffer(6)]],
    threadgroup half*          shmem  [[threadgroup(0)]], // 17408 bytes
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // Threadgroup tile: M=64 rows, N=32 cols
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // Shared Memory Layout (Double-Buffered):
    // Buffer size for A: [2][64][36] = 4608 halfs (padded to 36 stride to prevent bank conflicts) = 9216 bytes
    // Buffer size for B_gate: [2][32][32] = 2048 halfs = 4096 bytes
    // Buffer size for B_up:   [2][32][32] = 2048 halfs = 4096 bytes
    // Total shared memory = 17408 bytes
    threadgroup half (*sh_A)[64][36]      = (threadgroup half (*)[64][36])shmem;
    threadgroup half (*sh_B_gate)[32][32] = (threadgroup half (*)[32][32])(shmem + 4608);
    threadgroup half (*sh_B_up)[32][32]   = (threadgroup half (*)[32][32])(shmem + 4608 + 2048);

    // SIMDgroup Assignment:
    // SG0: Gate projection rows [0..31],  cols [0..31]
    // SG1: Gate projection rows [32..63], cols [0..31]
    // SG2: Up projection   rows [0..31],  cols [0..31]
    // SG3: Up projection   rows [32..63], cols [0..31]
    uint sg_row_offset = (simd_group_id & 1) * 32; // 0 for SG0/SG2, 32 for SG1/SG3
    bool is_gate_sg = (simd_group_id < 2);         // true for SG0/SG1, false for SG2/SG3

    // Accumulators: exactly 16 fragments of 8x8 (32 rows x 32 cols) using native hardware MMA FP32 accumulators
    simdgroup_matrix<float, 8, 8> acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint num_k_blocks = K / 32;

    // Helper: Cooperative load of X tile [64, 32] into sh_A[buf_idx] (shared by all 4 SIMDgroups)
    auto load_A_tile = [&](uint buf_idx, uint kb) {
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
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c]) = val.xy;
            *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c + 4]) = val.zw;
        }
    };

    // Helper: Cooperative load & dequantize 32 columns of B_gate and 32 columns of B_up
    auto dequant_B_tiles = [&](uint buf_idx, uint kb) {
        // Threads 0..31 dequantize Gate weight columns
        if (linear_tid < 32) {
            uint col = tg_col_start + linear_tid;
            if (col < N_mlp && kb < num_k_blocks) {
                block_q4_0 blk = B_gate[col * num_k_blocks + kb];
                half d = blk.d;
                half4 hd = half4(d);
                half4 h_off = half4(-8.0h * d);

                uint w0 = read_u32_unaligned(blk.qs + 0);
                uint w1 = read_u32_unaligned(blk.qs + 4);
                uint w2 = read_u32_unaligned(blk.qs + 8);
                uint w3 = read_u32_unaligned(blk.qs + 12);

                half4 vl[4], vh[4];
                vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
                vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
                vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
                vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
                vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sh_B_gate[buf_idx][i * 4 + 0][linear_tid] = vl[i][0];
                    sh_B_gate[buf_idx][i * 4 + 1][linear_tid] = vl[i][1];
                    sh_B_gate[buf_idx][i * 4 + 2][linear_tid] = vl[i][2];
                    sh_B_gate[buf_idx][i * 4 + 3][linear_tid] = vl[i][3];
                    sh_B_gate[buf_idx][16 + i * 4 + 0][linear_tid] = vh[i][0];
                    sh_B_gate[buf_idx][16 + i * 4 + 1][linear_tid] = vh[i][1];
                    sh_B_gate[buf_idx][16 + i * 4 + 2][linear_tid] = vh[i][2];
                    sh_B_gate[buf_idx][16 + i * 4 + 3][linear_tid] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B_gate[buf_idx][k][linear_tid] = 0.0h;
                }
            }
        }
        // Threads 32..63 dequantize Up weight columns
        else if (linear_tid >= 32 && linear_tid < 64) {
            uint up_col_idx = linear_tid - 32;
            uint col = tg_col_start + up_col_idx;
            if (col < N_mlp && kb < num_k_blocks) {
                block_q4_0 blk = B_up[col * num_k_blocks + kb];
                half d = blk.d;
                half4 hd = half4(d);
                half4 h_off = half4(-8.0h * d);

                uint w0 = read_u32_unaligned(blk.qs + 0);
                uint w1 = read_u32_unaligned(blk.qs + 4);
                uint w2 = read_u32_unaligned(blk.qs + 8);
                uint w3 = read_u32_unaligned(blk.qs + 12);

                half4 vl[4], vh[4];
                vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
                vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
                vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
                vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
                vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sh_B_up[buf_idx][i * 4 + 0][up_col_idx] = vl[i][0];
                    sh_B_up[buf_idx][i * 4 + 1][up_col_idx] = vl[i][1];
                    sh_B_up[buf_idx][i * 4 + 2][up_col_idx] = vl[i][2];
                    sh_B_up[buf_idx][i * 4 + 3][up_col_idx] = vl[i][3];
                    sh_B_up[buf_idx][16 + i * 4 + 0][up_col_idx] = vh[i][0];
                    sh_B_up[buf_idx][16 + i * 4 + 1][up_col_idx] = vh[i][1];
                    sh_B_up[buf_idx][16 + i * 4 + 2][up_col_idx] = vh[i][2];
                    sh_B_up[buf_idx][16 + i * 4 + 3][up_col_idx] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B_up[buf_idx][k][up_col_idx] = 0.0h;
                }
            }
        }
    };

    // Helper: Compute MMA on buffer buf_idx
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

    // Prologue: Load kb = 0 into buffer 0
    load_A_tile(0, 0);
    dequant_B_tiles(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Double-buffered main loop
    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur = kb & 1;
        uint nxt = cur ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            load_A_tile(nxt, next_kb);
            dequant_B_tiles(nxt, next_kb);
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

// ============================================================================
// 4. KERNEL 4: swiglu_mma_cooperative_64x64 (Full 64x64 Fused SwiGLU Engine)
// - 256-thread threadgroup (8 SIMDgroups)
// - [64, 64] Output Tile computed in a single unified dispatch
// - X activation loaded ONCE into [64, 36] SRAM and shared across all 8 SIMDgroups
// - SG0..SG3 compute Gate [64, 64] projection using 4-way 32x32 MMA sub-tiles
// - SG4..SG7 compute Up   [64, 64] projection using 4-way 32x32 MMA sub-tiles
// - Epilogue: In-SRAM Fused SwiGLU without touching DRAM for intermediate tensors
// ============================================================================
kernel void swiglu_mma_cooperative_64x64(
    device const half*         A      [[buffer(0)]], // [M, K]
    device const block_q4_0*   B_gate [[buffer(1)]], // [N_mlp, K/32]
    device const block_q4_0*   B_up   [[buffer(2)]], // [N_mlp, K/32]
    device half*               Out    [[buffer(3)]], // [M, N_mlp]
    constant uint&             M      [[buffer(4)]],
    constant uint&             N_mlp  [[buffer(5)]],
    constant uint&             K      [[buffer(6)]],
    threadgroup half*          shmem  [[threadgroup(0)]], // 25600 bytes
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint tg_row_start = tg_id.y * 64;
    uint tg_col_start = tg_id.x * 64;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint linear_tid = simd_group_id * 32 + simd_lane_id;

    // sh_A: [2][64][36] = 4608 halfs = 9216 bytes
    // sh_B_gate: [2][32][64] = 4096 halfs = 8192 bytes
    // sh_B_up:   [2][32][64] = 4096 halfs = 8192 bytes
    // Total = 25600 bytes
    threadgroup half (*sh_A)[64][36]      = (threadgroup half (*)[64][36])shmem;
    threadgroup half (*sh_B_gate)[32][64] = (threadgroup half (*)[32][64])(shmem + 4608);
    threadgroup half (*sh_B_up)[32][64]   = (threadgroup half (*)[32][64])(shmem + 4608 + 4096);

    // SIMDgroup mapping:
    // SG0..SG3: Gate projection tile [64, 64]
    //   SG0: r=[0..31],  c=[0..31]
    //   SG1: r=[0..31],  c=[32..63]
    //   SG2: r=[32..63], c=[0..31]
    //   SG3: r=[32..63], c=[32..63]
    // SG4..SG7: Up projection tile [64, 64]
    //   SG4: r=[0..31],  c=[0..31]
    //   SG5: r=[0..31],  c=[32..63]
    //   SG6: r=[32..63], c=[0..31]
    //   SG7: r=[32..63], c=[32..63]
    bool is_gate = (simd_group_id < 4);
    uint sub_sg_id = simd_group_id & 3; // 0..3
    uint sg_r = sub_sg_id / 2;          // 0 or 1
    uint sg_c = sub_sg_id % 2;          // 0 or 1
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

    auto load_A_tile = [&](uint buf_idx, uint kb) {
        // 256 threads load 64 rows x 32 cols (256 float4s)
        uint idx = linear_tid;
        uint r = idx / 4;
        uint c = (idx % 4) * 8;
        uint global_r = tg_row_start + r;
        uint global_c = kb * 32 + c;
        float4 val = float4(0.0f);
        if (global_r < M && global_c < K) {
            val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
        }
        *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c]) = val.xy;
        *reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c + 4]) = val.zw;
    };

    auto dequant_B_tiles = [&](uint buf_idx, uint kb) {
        // Threads 0..63 dequantize Gate (64 columns)
        if (linear_tid < 64) {
            uint col = tg_col_start + linear_tid;
            if (col < N_mlp && kb < num_k_blocks) {
                block_q4_0 blk = B_gate[col * num_k_blocks + kb];
                half d = blk.d;
                half4 hd = half4(d);
                half4 h_off = half4(-8.0h * d);

                uint w0 = read_u32_unaligned(blk.qs + 0);
                uint w1 = read_u32_unaligned(blk.qs + 4);
                uint w2 = read_u32_unaligned(blk.qs + 8);
                uint w3 = read_u32_unaligned(blk.qs + 12);

                half4 vl[4], vh[4];
                vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
                vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
                vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
                vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
                vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sh_B_gate[buf_idx][i * 4 + 0][linear_tid] = vl[i][0];
                    sh_B_gate[buf_idx][i * 4 + 1][linear_tid] = vl[i][1];
                    sh_B_gate[buf_idx][i * 4 + 2][linear_tid] = vl[i][2];
                    sh_B_gate[buf_idx][i * 4 + 3][linear_tid] = vl[i][3];
                    sh_B_gate[buf_idx][16 + i * 4 + 0][linear_tid] = vh[i][0];
                    sh_B_gate[buf_idx][16 + i * 4 + 1][linear_tid] = vh[i][1];
                    sh_B_gate[buf_idx][16 + i * 4 + 2][linear_tid] = vh[i][2];
                    sh_B_gate[buf_idx][16 + i * 4 + 3][linear_tid] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B_gate[buf_idx][k][linear_tid] = 0.0h;
                }
            }
        }
        // Threads 64..127 dequantize Up (64 columns)
        else if (linear_tid >= 64 && linear_tid < 128) {
            uint up_col_idx = linear_tid - 64;
            uint col = tg_col_start + up_col_idx;
            if (col < N_mlp && kb < num_k_blocks) {
                block_q4_0 blk = B_up[col * num_k_blocks + kb];
                half d = blk.d;
                half4 hd = half4(d);
                half4 h_off = half4(-8.0h * d);

                uint w0 = read_u32_unaligned(blk.qs + 0);
                uint w1 = read_u32_unaligned(blk.qs + 4);
                uint w2 = read_u32_unaligned(blk.qs + 8);
                uint w3 = read_u32_unaligned(blk.qs + 12);

                half4 vl[4], vh[4];
                vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
                vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
                vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
                vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
                vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
                vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sh_B_up[buf_idx][i * 4 + 0][up_col_idx] = vl[i][0];
                    sh_B_up[buf_idx][i * 4 + 1][up_col_idx] = vl[i][1];
                    sh_B_up[buf_idx][i * 4 + 2][up_col_idx] = vl[i][2];
                    sh_B_up[buf_idx][i * 4 + 3][up_col_idx] = vl[i][3];
                    sh_B_up[buf_idx][16 + i * 4 + 0][up_col_idx] = vh[i][0];
                    sh_B_up[buf_idx][16 + i * 4 + 1][up_col_idx] = vh[i][1];
                    sh_B_up[buf_idx][16 + i * 4 + 2][up_col_idx] = vh[i][2];
                    sh_B_up[buf_idx][16 + i * 4 + 3][up_col_idx] = vh[i][3];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sh_B_up[buf_idx][k][up_col_idx] = 0.0h;
                }
            }
        }
    };

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

            if (is_gate) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_load(b_frag[c], &sh_B_gate[buf_idx][k_off][sg_col_offset + c * 8], 64);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    simdgroup_load(b_frag[c], &sh_B_up[buf_idx][k_off][sg_col_offset + c * 8], 64);
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

    // Prologue
    load_A_tile(0, 0);
    dequant_B_tiles(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint cur = kb & 1;
        uint nxt = cur ^ 1;
        uint next_kb = kb + 1;

        if (next_kb < num_k_blocks) {
            load_A_tile(nxt, next_kb);
            dequant_B_tiles(nxt, next_kb);
        }

        compute_mma(cur);

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Epilogue: In-SRAM Fused SwiGLU
    // shmem has 32768 bytes (32 KB): sh_Gate[64][64] (16KB float = 8192 halfs) + sh_Up[64][64] (16KB float)
    threadgroup float (*sh_Gate)[64] = (threadgroup float (*)[64])shmem;
    threadgroup float (*sh_Up)[64]   = (threadgroup float (*)[64])(shmem + 8192);

    if (is_gate) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Gate[sg_row_offset + r * 8][sg_col_offset + c * 8], 64);
            }
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                simdgroup_store(acc[r][c], &sh_Up[sg_row_offset + r * 8][sg_col_offset + c * 8], 64);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fused On-Chip SwiGLU Activation & DRAM Writeback
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint elem_idx = linear_tid * 16 + i;
        uint r = elem_idx / 64;
        uint c = elem_idx % 64;
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

