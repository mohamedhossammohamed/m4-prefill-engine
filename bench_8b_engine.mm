#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <map>

// ============================================================================
// DATA STRUCTURES & QUANTIZATION TYPES
// ============================================================================
struct block_q4_0 {
    __fp16 d;
    uint8_t qs[16];
};

struct block_q8_0 {
    __fp16 d;
    int8_t qs[32];
};

// Deterministic PRNG
static uint32_t prng_state = 1337;
static inline float rand_uniform() {
    prng_state = prng_state * 1664525u + 1013904223u;
    return (float)prng_state / (float)0xFFFFFFFF;
}

void generate_activations(__fp16* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float u1 = std::max(1e-6f, rand_uniform());
        float u2 = rand_uniform();
        float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
        data[i] = (__fp16)(z0 * 0.35f);
    }
}

void generate_q4_0_weights(block_q4_0* blocks, size_t num_blocks, float scale_factor = 0.003f) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = (__fp16)(rand_uniform() * scale_factor + scale_factor * 0.1f);
        for (int i = 0; i < 16; i++) {
            uint8_t low = (uint8_t)(rand_uniform() * 16.0f);
            uint8_t high = (uint8_t)(rand_uniform() * 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

// Fast SiLU for float
static inline float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================================
// CPU GOLD REFERENCE FOR 8B PREFILL LAYER
// ============================================================================
struct CpuIntermediates {
    std::vector<float> Q_heads;
    std::vector<float> K_heads;
    std::vector<float> V_heads;
    std::vector<__fp16> O_attn;
    std::vector<float> O_proj;
    std::vector<__fp16> X_mid;
    std::vector<__fp16> S_mlp;
    std::vector<float> D_mlp;
};

void cpu_reference_prefill_layer(
    const __fp16* X_in,         // [M, K]
    const block_q4_0* W_q,      // [K, H*D]
    const block_q4_0* W_k,      // [K, H*D]
    const block_q4_0* W_v,      // [K, H*D]
    const block_q4_0* W_o,      // [H*D, K]
    const block_q4_0* W_gate,   // [K, N_mlp]
    const block_q4_0* W_up,     // [K, N_mlp]
    const block_q4_0* W_down,   // [N_mlp, K]
    __fp16* X_out,              // [M, K]
    uint32_t M,
    uint32_t K,
    uint32_t H,
    uint32_t D,
    uint32_t N_mlp,
    float attn_scale,
    CpuIntermediates* inter = nullptr)
{
    uint32_t attn_dim = H * D;

    // Multithreaded CPU GEMM: A[M, depth] @ B_q4_0[cols, depth/32] -> C[M, cols]
    auto cpu_gemm_q4_0 = [](const __fp16* A, const block_q4_0* B, float* C, uint32_t rows, uint32_t cols, uint32_t depth) {
        uint32_t nb = depth / 32;
        dispatch_apply(rows, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t r) {
            for (uint32_t c = 0; c < cols; c++) {
                const block_q4_0* b_col = B + c * nb;
                const __fp16* a_row = A + r * depth;
                float acc = 0.0f;
                for (uint32_t b = 0; b < nb; b++) {
                    float d = (float)b_col[b].d;
                    uint32_t a_off = b * 32;
                    for (int i = 0; i < 16; i++) {
                        uint8_t byte_val = b_col[b].qs[i];
                        int v0 = (int)(byte_val & 0x0F) - 8;
                        int v1 = (int)(byte_val >> 4) - 8;
                        acc += (float)a_row[a_off + i] * ((float)v0 * d);
                        acc += (float)a_row[a_off + i + 16] * ((float)v1 * d);
                    }
                }
                C[r * cols + c] = acc;
            }
        });
    };

    // 1. QKV Projections
    std::vector<float> Q_f(M * attn_dim), K_f(M * attn_dim), V_f(M * attn_dim);
    cpu_gemm_q4_0(X_in, W_q, Q_f.data(), M, attn_dim, K);
    cpu_gemm_q4_0(X_in, W_k, K_f.data(), M, attn_dim, K);
    cpu_gemm_q4_0(X_in, W_v, V_f.data(), M, attn_dim, K);

    // Convert Q, K, V to head-major [H, M, D]
    std::vector<float> Q_heads(H * M * D), K_heads(H * M * D), V_heads(H * M * D);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t d = 0; d < D; d++) {
                uint32_t src_idx = m * attn_dim + h * D + d;
                uint32_t dst_idx = (h * M + m) * D + d;
                Q_heads[dst_idx] = Q_f[src_idx];
                K_heads[dst_idx] = K_f[src_idx];
                V_heads[dst_idx] = V_f[src_idx];
            }
        }
    }

    // 2. Fused Causal Attention (Multithreaded over heads)
    std::vector<__fp16> O_attn(M * attn_dim);
    __fp16* o_attn_ptr = O_attn.data();
    const float* q_heads_ptr = Q_heads.data();
    const float* k_heads_ptr = K_heads.data();
    const float* v_heads_ptr = V_heads.data();

    dispatch_apply(H, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t h) {
        const float* q_h = q_heads_ptr + h * M * D;
        const float* k_h = k_heads_ptr + h * M * D;
        const float* v_h = v_heads_ptr + h * M * D;

        for (uint32_t i = 0; i < M; i++) {
            const float* q_row = q_h + i * D;
            std::vector<float> scores(i + 1);
            float max_s = -1e30f;

            for (uint32_t j = 0; j <= i; j++) {
                const float* k_row = k_h + j * D;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    dot += q_row[d] * k_row[d];
                }
                float s = dot * attn_scale;
                scores[j] = s;
                if (s > max_s) max_s = s;
            }

            float sum_exp = 0.0f;
            for (uint32_t j = 0; j <= i; j++) {
                scores[j] = std::exp(scores[j] - max_s);
                sum_exp += scores[j];
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            for (uint32_t d = 0; d < D; d++) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= i; j++) {
                    acc += scores[j] * v_h[j * D + d];
                }
                float o_val = acc * inv_sum;
                o_attn_ptr[i * attn_dim + h * D + d] = (__fp16)o_val;
            }
        }
    });

    // 3. Attention Output Projection & Residual
    std::vector<float> O_proj(M * K);
    cpu_gemm_q4_0(O_attn.data(), W_o, O_proj.data(), M, K, attn_dim);

    std::vector<__fp16> X_mid(M * K);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        X_mid[i] = (__fp16)((float)X_in[i] + O_proj[i]);
    }

    // 4. MLP Block: Gate & Up Projections
    std::vector<float> Gate_f(M * N_mlp), Up_f(M * N_mlp);
    cpu_gemm_q4_0(X_mid.data(), W_gate, Gate_f.data(), M, N_mlp, K);
    cpu_gemm_q4_0(X_mid.data(), W_up, Up_f.data(), M, N_mlp, K);

    // SwiGLU Activation: S = SiLU(Gate) * Up
    std::vector<__fp16> S_mlp(M * N_mlp);
    for (size_t i = 0; i < (size_t)M * N_mlp; i++) {
        float g = Gate_f[i];
        float u = Up_f[i];
        float swiglu = silu_f32(g) * u;
        S_mlp[i] = (__fp16)swiglu;
    }

    // Down Projection & Second Residual
    std::vector<float> Down_f(M * K);
    cpu_gemm_q4_0(S_mlp.data(), W_down, Down_f.data(), M, K, N_mlp);

    for (size_t i = 0; i < (size_t)M * K; i++) {
        X_out[i] = (__fp16)((float)X_mid[i] + Down_f[i]);
    }

    if (inter) {
        inter->Q_heads = std::move(Q_heads);
        inter->K_heads = std::move(K_heads);
        inter->V_heads = std::move(V_heads);
        inter->O_attn = std::move(O_attn);
        inter->O_proj = std::move(O_proj);
        inter->X_mid = std::move(X_mid);
        inter->S_mlp = std::move(S_mlp);
        inter->D_mlp = std::move(Down_f);
    }
}

// ============================================================================
// TIMING & PROFILING METRICS
// ============================================================================
struct LayerProfile {
    double qkv_ms;
    double attn_ms;
    double o_proj_ms;
    double mlp_ms;
    double total_ms;
    double throughput_tok_s;
    double full_model_32l_ms;
    double full_model_throughput_tok_s;
    double dram_bandwidth_gb_s;
};

// ============================================================================
// MAIN UNIFIED 8B PREFILL ENGINE BENCHMARK
// ============================================================================
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "        J.A.R.V.I.S. UNIFIED END-TO-END 8B PREFILL ENGINE BENCHMARK (APPLE M4)           " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[-] Error: Metal device initialization failed." << std::endl;
            return 1;
        }

        std::cout << "[+] Hardware Device: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"unified_8b_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[-] Error loading unified_8b_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[-] Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // ARCHITECTURE CONSTANTS (LLaMA-3 8B STANDARD)
        // --------------------------------------------------------------------
        const uint32_t K = 4096;         // Hidden Dimension
        const uint32_t H = 32;           // Attention Heads
        const uint32_t D = 128;          // Head Dimension
        const uint32_t ATTN_DIM = H * D; // 4096
        const uint32_t N_MLP = 14336;    // Intermediate MLP Dimension
        const float attn_scale = 1.0f / std::sqrt((float)D); // 1.0f / sqrt(128) ≈ 0.0883883
        const uint32_t NUM_LAYERS = 32;  // Full 8B model layers

        // Calculate layer weight bytes (Q4_0: 18 bytes per 32 weights = 4.5 bits/weight)
        size_t qkv_blocks      = (size_t)ATTN_DIM * (K / 32);       // 4096 * 128 = 524,288
        size_t o_blocks        = (size_t)K * (ATTN_DIM / 32);       // 4096 * 128 = 524,288
        size_t mlp_up_blocks   = (size_t)N_MLP * (K / 32);          // 14336 * 128 = 1,835,008
        size_t mlp_down_blocks = (size_t)K * (N_MLP / 32);          // 4096 * 448 = 1,835,008

        size_t total_weight_blocks_layer = 3 * qkv_blocks + o_blocks + 2 * mlp_up_blocks + mlp_down_blocks;
        size_t total_weight_bytes_layer  = total_weight_blocks_layer * sizeof(block_q4_0);
        double layer_mb = (double)total_weight_bytes_layer / (1024.0 * 1024.0);
        double full_model_gb = (double)(total_weight_bytes_layer * NUM_LAYERS) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "[+] Architecture Configuration (LLaMA-3 8B):" << std::endl;
        std::cout << "    - Hidden Dimension (K):          " << K << std::endl;
        std::cout << "    - Attention Heads (H):           " << H << std::endl;
        std::cout << "    - Head Dimension (D):            " << D << " (Attn Dim = " << ATTN_DIM << ")" << std::endl;
        std::cout << "    - MLP Intermediate Dimension:    " << N_MLP << std::endl;
        std::cout << "    - Single Layer Q4_0 Weights:     " << std::fixed << std::setprecision(2) << layer_mb << " MB" << std::endl;
        std::cout << "    - Full Model (32 Layers) Weights:" << std::fixed << std::setprecision(2) << full_model_gb << " GB" << std::endl;
        std::cout << "    - L2/SLC Cache Capacity:         ~24 MB (Weights strictly spill to Unified DRAM!)" << std::endl;

        // --------------------------------------------------------------------
        // LOAD METAL PIPELINES
        // --------------------------------------------------------------------
        auto load_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [library newFunctionWithName:name];
            if (!fn) {
                std::cerr << "[-] Error: Function not found: " << [name UTF8String] << std::endl;
                exit(1);
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (error) {
                std::cerr << "[-] Error compiling PSO for " << [name UTF8String] << ": "
                          << [[error localizedDescription] UTF8String] << std::endl;
                exit(1);
            }
            return pso;
        };

        // Optimized Engine Pipelines
        id<MTLComputePipelineState> pso_pipe_gemm_32x32     = load_pso(@"pipe_gemm_q4_0_32x32");
        id<MTLComputePipelineState> pso_pipe_qkv_head       = load_pso(@"pipe_qkv_head_gemm_q4_0");
        id<MTLComputePipelineState> pso_fused_gate_up       = load_pso(@"fused_gate_up_swiglu_q4_0");
        id<MTLComputePipelineState> pso_flash_attn_fp16     = load_pso(@"flash_attn_fp16_causal_d128");
        id<MTLComputePipelineState> pso_flash_attn_q8_0     = load_pso(@"flash_attn_q8_0_causal_d128");
        id<MTLComputePipelineState> pso_quantize_kv_q8_0    = load_pso(@"quantize_kv_to_q8_0");
        id<MTLComputePipelineState> pso_swiglu              = load_pso(@"swiglu_activation");
        id<MTLComputePipelineState> pso_residual_add        = load_pso(@"vector_add_residual");
        id<MTLComputePipelineState> pso_transpose_hd        = load_pso(@"transpose_m_hd_to_h_m_d");
        id<MTLComputePipelineState> pso_transpose_h_m_d     = load_pso(@"transpose_h_m_d_to_m_hd");

        // Baseline Engine Pipelines
        id<MTLComputePipelineState> pso_llamacpp_gemm       = load_pso(@"llamacpp_style_mul_mm_q4_0");
        id<MTLComputePipelineState> pso_naive_qk            = load_pso(@"naive_attn_qk_causal");
        id<MTLComputePipelineState> pso_naive_softmax       = load_pso(@"naive_attn_softmax");
        id<MTLComputePipelineState> pso_naive_pv            = load_pso(@"naive_attn_pv");

        // --------------------------------------------------------------------
        // ALLOCATE 8B MODEL WEIGHTS IN Q4_0 FORMAT
        // --------------------------------------------------------------------
        std::vector<block_q4_0> h_W_q(qkv_blocks);
        std::vector<block_q4_0> h_W_k(qkv_blocks);
        std::vector<block_q4_0> h_W_v(qkv_blocks);
        std::vector<block_q4_0> h_W_o(o_blocks);
        std::vector<block_q4_0> h_W_gate(mlp_up_blocks);
        std::vector<block_q4_0> h_W_up(mlp_up_blocks);
        std::vector<block_q4_0> h_W_down(mlp_down_blocks);

        generate_q4_0_weights(h_W_q.data(), qkv_blocks);
        generate_q4_0_weights(h_W_k.data(), qkv_blocks);
        generate_q4_0_weights(h_W_v.data(), qkv_blocks);
        generate_q4_0_weights(h_W_o.data(), o_blocks);
        generate_q4_0_weights(h_W_gate.data(), mlp_up_blocks);
        generate_q4_0_weights(h_W_up.data(), mlp_up_blocks);
        generate_q4_0_weights(h_W_down.data(), mlp_down_blocks);

        id<MTLBuffer> d_W_q    = [device newBufferWithBytes:h_W_q.data() length:qkv_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_k    = [device newBufferWithBytes:h_W_k.data() length:qkv_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_v    = [device newBufferWithBytes:h_W_v.data() length:qkv_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_o    = [device newBufferWithBytes:h_W_o.data() length:o_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_gate = [device newBufferWithBytes:h_W_gate.data() length:mlp_up_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_up   = [device newBufferWithBytes:h_W_up.data() length:mlp_up_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_W_down = [device newBufferWithBytes:h_W_down.data() length:mlp_down_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];

        // ====================================================================
        // STAGE 1: NUMERICAL ACCURACY VERIFICATION (GPU vs CPU GOLD REFERENCE)
        // ====================================================================
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 1] END-TO-END 8B LAYER NUMERICAL VERIFICATION (GPU vs CPU REFERENCE)" << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const std::vector<uint32_t> verify_lengths = {128, 512, 1024};
        for (uint32_t M : verify_lengths) {
            std::cout << "\n--- Verifying 8B Layer at Sequence Length M = " << M << " tokens ---" << std::endl;

            // Generate synthetic activations
            size_t in_elements = (size_t)M * K;
            std::vector<__fp16> h_X_in(in_elements);
            std::vector<__fp16> h_X_out_cpu(in_elements);
            std::vector<__fp16> h_X_out_gpu_opt(in_elements);
            std::vector<__fp16> h_X_out_gpu_q8(in_elements);

            generate_activations(h_X_in.data(), in_elements);

            CpuIntermediates cpu_inter;
            // Run CPU Ground Truth
            std::cout << "    [+] Running CPU Gold Reference forward pass..." << std::flush;
            auto t0_cpu = std::chrono::high_resolution_clock::now();
            cpu_reference_prefill_layer(
                h_X_in.data(), h_W_q.data(), h_W_k.data(), h_W_v.data(), h_W_o.data(),
                h_W_gate.data(), h_W_up.data(), h_W_down.data(), h_X_out_cpu.data(),
                M, K, H, D, N_MLP, attn_scale, &cpu_inter);
            auto t1_cpu = std::chrono::high_resolution_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();
            std::cout << " Done (" << std::fixed << std::setprecision(2) << cpu_ms << " ms)" << std::endl;

            // Allocate GPU activation buffers
            id<MTLBuffer> d_X_in     = [device newBufferWithBytes:h_X_in.data() length:in_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_Q        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O_attn   = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O_proj   = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_X_mid    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_S_mlp    = [device newBufferWithLength:(size_t)M * N_MLP * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_D_mlp    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_X_out    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];

            // ----------------------------------------------------------------
            // 1. Run Optimized Unified Prefill Layer (FP16 KV Cache)
            // ----------------------------------------------------------------
            {
                id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                // Stage A: Direct Head QKV Projections [M, K] -> [H, M, D]
                uint32_t qkv_N = ATTN_DIM;
                NSUInteger tg_x = (qkv_N + 31) / 32;
                NSUInteger tg_y = (M + 31) / 32;
                MTLSize grid_qkv = MTLSizeMake(tg_x, tg_y, 1);
                MTLSize tg_size_32 = MTLSizeMake(32, 1, 1);

                // Q Projection
                [enc setComputePipelineState:pso_pipe_qkv_head];
                [enc setBuffer:d_X_in offset:0 atIndex:0];
                [enc setBuffer:d_W_q offset:0 atIndex:1];
                [enc setBuffer:d_Q offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                // K Projection
                [enc setBuffer:d_W_k offset:0 atIndex:1];
                [enc setBuffer:d_K offset:0 atIndex:2];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                // V Projection
                [enc setBuffer:d_W_v offset:0 atIndex:1];
                [enc setBuffer:d_V offset:0 atIndex:2];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                // Stage B: Fused FlashAttention FP16 (32x16 Tile) -> [M, H*D]
                [enc setComputePipelineState:pso_flash_attn_fp16];
                [enc setBuffer:d_Q offset:0 atIndex:0];
                [enc setBuffer:d_K offset:0 atIndex:1];
                [enc setBuffer:d_V offset:0 atIndex:2];
                [enc setBuffer:d_O_attn offset:0 atIndex:3];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&attn_scale length:sizeof(float) atIndex:6];
                [enc setThreadgroupMemoryLength:16384 atIndex:0];
                MTLSize grid_fa = MTLSizeMake((M + 31) / 32, H, 1);
                [enc dispatchThreadgroups:grid_fa threadsPerThreadgroup:tg_size_32];

                // Stage C: O-Projection [M, ATTN_DIM] -> [M, K]
                [enc setComputePipelineState:pso_pipe_gemm_32x32];
                [enc setBuffer:d_O_attn offset:0 atIndex:0];
                [enc setBuffer:d_W_o offset:0 atIndex:1];
                [enc setBuffer:d_O_proj offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:5];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_o = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_o threadsPerThreadgroup:tg_size_32];

                // Residual Addition: X_mid = X_in + O_proj
                uint32_t num_f4 = (uint32_t)(in_elements / 4);
                [enc setComputePipelineState:pso_residual_add];
                [enc setBuffer:d_X_in offset:0 atIndex:0];
                [enc setBuffer:d_O_proj offset:0 atIndex:1];
                [enc setBuffer:d_X_mid offset:0 atIndex:2];
                [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // Stage D: Fused Gate+Up GEMM with In-Kernel SwiGLU Epilogue
                [enc setComputePipelineState:pso_fused_gate_up];
                [enc setBuffer:d_X_mid offset:0 atIndex:0];
                [enc setBuffer:d_W_gate offset:0 atIndex:1];
                [enc setBuffer:d_W_up offset:0 atIndex:2];
                [enc setBuffer:d_S_mlp offset:0 atIndex:3];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_mlp_up = MTLSizeMake((N_MLP + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_mlp_up threadsPerThreadgroup:tg_size_32];

                // Stage E: Down-Projection [M, N_MLP] -> [M, K]
                [enc setComputePipelineState:pso_pipe_gemm_32x32];
                [enc setBuffer:d_S_mlp offset:0 atIndex:0];
                [enc setBuffer:d_W_down offset:0 atIndex:1];
                [enc setBuffer:d_D_mlp offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_down = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_down threadsPerThreadgroup:tg_size_32];

                // Final Residual Addition: X_out = X_mid + D_mlp
                [enc setComputePipelineState:pso_residual_add];
                [enc setBuffer:d_X_mid offset:0 atIndex:0];
                [enc setBuffer:d_D_mlp offset:0 atIndex:1];
                [enc setBuffer:d_X_out offset:0 atIndex:2];
                [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                memcpy(h_X_out_gpu_opt.data(), [d_X_out contents], in_elements * sizeof(__fp16));
            }

            // ----------------------------------------------------------------
            // 2. Run Optimized Unified Prefill Layer (Q8_0 KV Cache)
            // ----------------------------------------------------------------
            {
                size_t num_kv_blocks = (size_t)H * M * (D / 32); // D=128 => 4 blocks per token
                id<MTLBuffer> d_K_q8 = [device newBufferWithLength:num_kv_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_V_q8 = [device newBufferWithLength:num_kv_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                // QKV Projections
                uint32_t qkv_N = ATTN_DIM;
                NSUInteger tg_x = (qkv_N + 31) / 32;
                NSUInteger tg_y = (M + 31) / 32;
                MTLSize grid_qkv = MTLSizeMake(tg_x, tg_y, 1);
                MTLSize tg_size_32 = MTLSizeMake(32, 1, 1);

                [enc setComputePipelineState:pso_pipe_qkv_head];
                [enc setBuffer:d_X_in offset:0 atIndex:0];
                [enc setBuffer:d_W_q offset:0 atIndex:1];
                [enc setBuffer:d_Q offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                [enc setBuffer:d_W_k offset:0 atIndex:1];
                [enc setBuffer:d_K offset:0 atIndex:2];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                [enc setBuffer:d_W_v offset:0 atIndex:1];
                [enc setBuffer:d_V offset:0 atIndex:2];
                [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                // Quantize K & V to Q8_0 on-the-fly
                uint32_t total_kv_blocks = (uint32_t)num_kv_blocks;
                [enc setComputePipelineState:pso_quantize_kv_q8_0];
                [enc setBuffer:d_K offset:0 atIndex:0];
                [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                [enc setBytes:&total_kv_blocks length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(total_kv_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:d_V offset:0 atIndex:0];
                [enc setBuffer:d_V_q8 offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(total_kv_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // FlashAttention Q8_0 (32x16 Tile for D=128)
                [enc setComputePipelineState:pso_flash_attn_q8_0];
                [enc setBuffer:d_Q offset:0 atIndex:0];
                [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                [enc setBuffer:d_O_attn offset:0 atIndex:3];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&attn_scale length:sizeof(float) atIndex:6];
                [enc setThreadgroupMemoryLength:16384 atIndex:0];
                MTLSize grid_fa_q8 = MTLSizeMake((M + 31) / 32, H, 1);
                [enc dispatchThreadgroups:grid_fa_q8 threadsPerThreadgroup:tg_size_32];

                // O-Projection & Residual
                [enc setComputePipelineState:pso_pipe_gemm_32x32];
                [enc setBuffer:d_O_attn offset:0 atIndex:0];
                [enc setBuffer:d_W_o offset:0 atIndex:1];
                [enc setBuffer:d_O_proj offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:5];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_o = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_o threadsPerThreadgroup:tg_size_32];

                uint32_t num_f4 = (uint32_t)(in_elements / 4);
                [enc setComputePipelineState:pso_residual_add];
                [enc setBuffer:d_X_in offset:0 atIndex:0];
                [enc setBuffer:d_O_proj offset:0 atIndex:1];
                [enc setBuffer:d_X_mid offset:0 atIndex:2];
                [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // MLP Gate/Up + SwiGLU + Down + Residual
                [enc setComputePipelineState:pso_fused_gate_up];
                [enc setBuffer:d_X_mid offset:0 atIndex:0];
                [enc setBuffer:d_W_gate offset:0 atIndex:1];
                [enc setBuffer:d_W_up offset:0 atIndex:2];
                [enc setBuffer:d_S_mlp offset:0 atIndex:3];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_mlp_up = MTLSizeMake((N_MLP + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_mlp_up threadsPerThreadgroup:tg_size_32];

                [enc setComputePipelineState:pso_pipe_gemm_32x32];
                [enc setBuffer:d_S_mlp offset:0 atIndex:0];
                [enc setBuffer:d_W_down offset:0 atIndex:1];
                [enc setBuffer:d_D_mlp offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                [enc setThreadgroupMemoryLength:4096 atIndex:0];
                MTLSize grid_down = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                [enc dispatchThreadgroups:grid_down threadsPerThreadgroup:tg_size_32];

                [enc setComputePipelineState:pso_residual_add];
                [enc setBuffer:d_X_mid offset:0 atIndex:0];
                [enc setBuffer:d_D_mlp offset:0 atIndex:1];
                [enc setBuffer:d_X_out offset:0 atIndex:2];
                [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                memcpy(h_X_out_gpu_q8.data(), [d_X_out contents], in_elements * sizeof(__fp16));
            }

            // ----------------------------------------------------------------
            // Compute Verification Metrics
            // ----------------------------------------------------------------
            auto compare_bufs = [](const __fp16* a, const __fp16* b, size_t count, const std::string& label) {
                float max_d = 0.0f;
                float sum_d = 0.0f;
                for (size_t i = 0; i < count; i++) {
                    float va = (float)a[i];
                    float vb = (float)b[i];
                    if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                        fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                        assert(false && "Numerical instability detected (NaN/Inf)");
                        exit(1);
                    }
                    float d = std::fabs(va - vb);
                    if (d > max_d) max_d = d;
                    sum_d += d;
                }
                std::cout << "      -> [" << label << "] MaxDiff: " << max_d << ", AvgDiff: " << (sum_d / count)
                          << " (Sample GPU: " << (float)a[0] << ", CPU: " << (float)b[0] << ")" << std::endl;
            };

            std::cout << "\n    [*] Diagnostic Stage Comparison for M = " << M << ":" << std::endl;
            std::vector<__fp16> Q_cpu_fp16(H * M * D);
            for (size_t i = 0; i < H * M * D; i++) Q_cpu_fp16[i] = (__fp16)cpu_inter.Q_heads[i];
            compare_bufs((const __fp16*)[d_Q contents], Q_cpu_fp16.data(), H * M * D, "Q Matrix [H, M, D]");
            compare_bufs((const __fp16*)[d_O_attn contents], cpu_inter.O_attn.data(), M * ATTN_DIM, "O_attn Matrix [M, ATTN_DIM]");
            compare_bufs((const __fp16*)[d_X_mid contents], cpu_inter.X_mid.data(), M * K, "X_mid Residual [M, K]");
            compare_bufs((const __fp16*)[d_S_mlp contents], cpu_inter.S_mlp.data(), M * N_MLP, "S_mlp SwiGLU [M, N_MLP]");
            compare_bufs((const __fp16*)[d_X_out contents], h_X_out_cpu.data(), M * K, "X_out Final [M, K]");

            auto eval_diff = [&](const std::vector<__fp16>& gpu, const std::string& name) {
                float max_diff = 0.0f;
                float sum_diff = 0.0f;
                float sum_sq = 0.0f;
                for (size_t i = 0; i < in_elements; i++) {
                    float va = (float)gpu[i];
                    float vb = (float)h_X_out_cpu[i];
                    if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                        fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                        assert(false && "Numerical instability detected (NaN/Inf)");
                        exit(1);
                    }
                    float diff = std::fabs(va - vb);
                    if (diff > max_diff) max_diff = diff;
                    sum_diff += diff;
                    sum_sq += diff * diff;
                }
                float avg_diff = sum_diff / in_elements;
                float rmse = std::sqrt(sum_sq / in_elements);

                bool pass = (max_diff <= 0.05f);
                std::cout << "    [" << (pass ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << name
                          << " | MaxDiff: " << std::fixed << std::setprecision(5) << max_diff
                          << " | AvgDiff: " << avg_diff
                          << " | RMSE: " << rmse
                          << " | Threshold: <= 0.05" << std::endl;
                assert(pass && "Numerical assertion failed!");
            };

            eval_diff(h_X_out_gpu_opt, "Unified 8B Engine (FP16 KV)");
            eval_diff(h_X_out_gpu_q8,  "Unified 8B Engine (Q8_0 KV)");
        }

        std::cout << "\n[✓] 100% NUMERICAL ACCURACY CONFIRMED AT 8B SCALE ACROSS ALL SUITES (MaxDiff <= 0.05)" << std::endl;

        // ====================================================================
        // STAGE 2: DETAILED BENCHMARKING & BREAKDOWN ACROSS ALL SEQUENCE LENGTHS
        // ====================================================================
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 2] 8B END-TO-END PREFILL BENCHMARK & COMPONENT BREAKDOWN" << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const std::vector<uint32_t> benchmark_lengths = {33, 127, 128, 129, 512, 1023, 1024, 2047, 2048};

        struct SummaryRow {
            uint32_t M;
            double baseline_ms;
            double unified_fp16_ms;
            double unified_q8_ms;
            double speedup_fp16;
            double speedup_q8;
            double tok_s_unified_fp16;
            double tok_s_unified_q8;
            double full_32l_fp16_ms;
            double full_32l_q8_ms;
            double full_32l_tok_s;
            double dram_bw_gb_s;
        };

        std::vector<SummaryRow> summary_table;

        for (uint32_t M : benchmark_lengths) {
            std::cout << "\n------------------------------------------------------------------------------------------" << std::endl;
            std::cout << ">>> BENCHMARKING 8B SCALE AT SEQUENCE LENGTH M = " << M << " TOKENS (Hidden K=" << K << ", H=" << H << ", D=" << D << ")" << std::endl;
            std::cout << "------------------------------------------------------------------------------------------" << std::endl;

            size_t in_elements = (size_t)M * K;
            std::vector<__fp16> h_X(in_elements);
            generate_activations(h_X.data(), in_elements);

            id<MTLBuffer> d_X_in     = [device newBufferWithBytes:h_X.data() length:in_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_Q        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V        = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O_attn   = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O_proj   = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_X_mid    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_S_mlp    = [device newBufferWithLength:(size_t)M * N_MLP * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_D_mlp    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_X_out    = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];

            size_t num_kv_blocks = (size_t)H * M * (D / 32);
            id<MTLBuffer> d_K_q8 = [device newBufferWithLength:num_kv_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V_q8 = [device newBufferWithLength:num_kv_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];

            // Intermediate buffers for baseline
            id<MTLBuffer> d_Q_raw    = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K_raw    = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V_raw    = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_S_matrix = [device newBufferWithLength:(size_t)H * M * M * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_P_matrix = [device newBufferWithLength:(size_t)H * M * M * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_Gate_raw = [device newBufferWithLength:(size_t)M * N_MLP * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_Up_raw   = [device newBufferWithLength:(size_t)M * N_MLP * sizeof(__fp16) options:MTLResourceStorageModeShared];

            const int WARMUP_ITERS = 5;
            const int BENCH_ITERS  = 15;

            // ----------------------------------------------------------------
            // 1. BASELINE LAYER PIPELINE EXECUTION (LLAMA.CPP STYLE)
            // ----------------------------------------------------------------
            auto run_baseline_pass = [&](LayerProfile& prof) {
                auto time_encoder = [&](void (^encode_ops)(id<MTLComputeCommandEncoder>)) -> double {
                    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    encode_ops(enc);
                    [enc endEncoding];
                    __block CFTimeInterval gpuStart = 0;
                    __block CFTimeInterval gpuEnd = 0;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        gpuStart = buffer.GPUStartTime;
                        gpuEnd = buffer.GPUEndTime;
                    }];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return (gpuEnd - gpuStart) * 1000.0;
                };

                // Component A: QKV Projections (llama.cpp GEMM)
                double t_qkv = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:pso_llamacpp_gemm];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    [enc setThreadgroupMemoryLength:2048 atIndex:1];
                    MTLSize grid = MTLSizeMake((ATTN_DIM + 31) / 32, (M + 63) / 64, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);

                    // Q
                    [enc setBuffer:d_X_in offset:0 atIndex:0];
                    [enc setBuffer:d_W_q offset:0 atIndex:1];
                    [enc setBuffer:d_Q_raw offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

                    // K
                    [enc setBuffer:d_W_k offset:0 atIndex:1];
                    [enc setBuffer:d_K_raw offset:0 atIndex:2];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

                    // V
                    [enc setBuffer:d_W_v offset:0 atIndex:1];
                    [enc setBuffer:d_V_raw offset:0 atIndex:2];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

                    // Transpose to [H, M, D]
                    [enc setComputePipelineState:pso_transpose_hd];
                    [enc setBuffer:d_Q_raw offset:0 atIndex:0];
                    [enc setBuffer:d_Q offset:0 atIndex:1];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
                    [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                    [enc setBuffer:d_K_raw offset:0 atIndex:0];
                    [enc setBuffer:d_K offset:0 atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                    [enc setBuffer:d_V_raw offset:0 atIndex:0];
                    [enc setBuffer:d_V offset:0 atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                });

                // Component B: 3-Stage Baseline Attention
                double t_attn = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    // QK Causal
                    [enc setComputePipelineState:pso_naive_qk];
                    [enc setBuffer:d_Q offset:0 atIndex:0];
                    [enc setBuffer:d_K offset:0 atIndex:1];
                    [enc setBuffer:d_S_matrix offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&attn_scale length:sizeof(float) atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(M, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                    // Softmax
                    [enc setComputePipelineState:pso_naive_softmax];
                    [enc setBuffer:d_S_matrix offset:0 atIndex:0];
                    [enc setBuffer:d_P_matrix offset:0 atIndex:1];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    // PV
                    [enc setComputePipelineState:pso_naive_pv];
                    [enc setBuffer:d_P_matrix offset:0 atIndex:0];
                    [enc setBuffer:d_V offset:0 atIndex:1];
                    [enc setBuffer:d_Q offset:0 atIndex:2]; // reuse temp buffer [H, M, D]
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(D, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                });

                // Component C: O-Proj & Residual
                double t_oproj = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    // Reformat to [M, ATTN_DIM]
                    [enc setComputePipelineState:pso_transpose_h_m_d];
                    [enc setBuffer:d_Q offset:0 atIndex:0];
                    [enc setBuffer:d_O_attn offset:0 atIndex:1];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
                    [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

                    // llama.cpp GEMM for O-proj
                    [enc setComputePipelineState:pso_llamacpp_gemm];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    [enc setThreadgroupMemoryLength:2048 atIndex:1];
                    MTLSize grid = MTLSizeMake((K + 31) / 32, (M + 63) / 64, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);
                    [enc setBuffer:d_O_attn offset:0 atIndex:0];
                    [enc setBuffer:d_W_o offset:0 atIndex:1];
                    [enc setBuffer:d_O_proj offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:5];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

                    // Residual Add
                    uint32_t num_f4 = (uint32_t)(in_elements / 4);
                    [enc setComputePipelineState:pso_residual_add];
                    [enc setBuffer:d_X_in offset:0 atIndex:0];
                    [enc setBuffer:d_O_proj offset:0 atIndex:1];
                    [enc setBuffer:d_X_mid offset:0 atIndex:2];
                    [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });

                // Component D & E: MLP Block (Gate, Up, SwiGLU, Down, Residual)
                double t_mlp = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:pso_llamacpp_gemm];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    [enc setThreadgroupMemoryLength:2048 atIndex:1];
                    MTLSize grid_mlp = MTLSizeMake((N_MLP + 31) / 32, (M + 63) / 64, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);

                    // Gate
                    [enc setBuffer:d_X_mid offset:0 atIndex:0];
                    [enc setBuffer:d_W_gate offset:0 atIndex:1];
                    [enc setBuffer:d_Gate_raw offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
                    [enc dispatchThreadgroups:grid_mlp threadsPerThreadgroup:tg];

                    // Up
                    [enc setBuffer:d_W_up offset:0 atIndex:1];
                    [enc setBuffer:d_Up_raw offset:0 atIndex:2];
                    [enc dispatchThreadgroups:grid_mlp threadsPerThreadgroup:tg];

                    // Standalone SwiGLU
                    uint32_t mlp_elems = M * N_MLP;
                    [enc setComputePipelineState:pso_swiglu];
                    [enc setBuffer:d_Gate_raw offset:0 atIndex:0];
                    [enc setBuffer:d_Up_raw offset:0 atIndex:1];
                    [enc setBuffer:d_S_mlp offset:0 atIndex:2];
                    [enc setBytes:&mlp_elems length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake((mlp_elems + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Down
                    [enc setComputePipelineState:pso_llamacpp_gemm];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    [enc setThreadgroupMemoryLength:2048 atIndex:1];
                    MTLSize grid_down = MTLSizeMake((K + 31) / 32, (M + 63) / 64, 1);
                    [enc setBuffer:d_S_mlp offset:0 atIndex:0];
                    [enc setBuffer:d_W_down offset:0 atIndex:1];
                    [enc setBuffer:d_D_mlp offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                    [enc dispatchThreadgroups:grid_down threadsPerThreadgroup:tg];

                    // Residual
                    uint32_t num_f4 = (uint32_t)(in_elements / 4);
                    [enc setComputePipelineState:pso_residual_add];
                    [enc setBuffer:d_X_mid offset:0 atIndex:0];
                    [enc setBuffer:d_D_mlp offset:0 atIndex:1];
                    [enc setBuffer:d_X_out offset:0 atIndex:2];
                    [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });

                prof.qkv_ms = t_qkv;
                prof.attn_ms = t_attn;
                prof.o_proj_ms = t_oproj;
                prof.mlp_ms = t_mlp;
                prof.total_ms = t_qkv + t_attn + t_oproj + t_mlp;
                prof.throughput_tok_s = ((double)M / (prof.total_ms * 1e-3));
                prof.full_model_32l_ms = prof.total_ms * NUM_LAYERS;
                prof.full_model_throughput_tok_s = ((double)M / (prof.full_model_32l_ms * 1e-3));
                prof.dram_bandwidth_gb_s = ((double)total_weight_bytes_layer / (prof.total_ms * 1e-3)) / 1e9;
            };

            // ----------------------------------------------------------------
            // 2. OPTIMIZED UNIFIED PREFILL PIPELINE (FP16 & Q8_0)
            // ----------------------------------------------------------------
            auto run_unified_pass = [&](bool is_q8, LayerProfile& prof) {
                auto time_encoder = [&](void (^encode_ops)(id<MTLComputeCommandEncoder>)) -> double {
                    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    encode_ops(enc);
                    [enc endEncoding];
                    __block CFTimeInterval gpuStart = 0;
                    __block CFTimeInterval gpuEnd = 0;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        gpuStart = buffer.GPUStartTime;
                        gpuEnd = buffer.GPUEndTime;
                    }];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return (gpuEnd - gpuStart) * 1000.0;
                };

                MTLSize tg_size_32 = MTLSizeMake(32, 1, 1);

                // Stage A: QKV Projections (Direct Head Layout)
                double t_qkv = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    uint32_t qkv_N = ATTN_DIM;
                    NSUInteger tg_x = (qkv_N + 31) / 32;
                    NSUInteger tg_y = (M + 31) / 32;
                    MTLSize grid_qkv = MTLSizeMake(tg_x, tg_y, 1);

                    [enc setComputePipelineState:pso_pipe_qkv_head];
                    [enc setBuffer:d_X_in offset:0 atIndex:0];
                    [enc setBuffer:d_W_q offset:0 atIndex:1];
                    [enc setBuffer:d_Q offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                    [enc setBuffer:d_W_k offset:0 atIndex:1];
                    [enc setBuffer:d_K offset:0 atIndex:2];
                    [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                    [enc setBuffer:d_W_v offset:0 atIndex:1];
                    [enc setBuffer:d_V offset:0 atIndex:2];
                    [enc dispatchThreadgroups:grid_qkv threadsPerThreadgroup:tg_size_32];

                    if (is_q8) {
                        uint32_t total_kv_blocks = (uint32_t)num_kv_blocks;
                        [enc setComputePipelineState:pso_quantize_kv_q8_0];
                        [enc setBuffer:d_K offset:0 atIndex:0];
                        [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                        [enc setBytes:&total_kv_blocks length:sizeof(uint32_t) atIndex:2];
                        [enc dispatchThreads:MTLSizeMake(total_kv_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc setBuffer:d_V offset:0 atIndex:0];
                        [enc setBuffer:d_V_q8 offset:0 atIndex:1];
                        [enc dispatchThreads:MTLSizeMake(total_kv_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                });

                // Stage B: Fused FlashAttention Engine for D=128
                double t_attn = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    if (!is_q8) {
                        [enc setComputePipelineState:pso_flash_attn_fp16];
                        [enc setBuffer:d_Q offset:0 atIndex:0];
                        [enc setBuffer:d_K offset:0 atIndex:1];
                        [enc setBuffer:d_V offset:0 atIndex:2];
                        [enc setBuffer:d_O_attn offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
                        [enc setBytes:&attn_scale length:sizeof(float) atIndex:6];
                        [enc setThreadgroupMemoryLength:16384 atIndex:0];
                        MTLSize grid_fa = MTLSizeMake((M + 31) / 32, H, 1);
                        [enc dispatchThreadgroups:grid_fa threadsPerThreadgroup:tg_size_32];
                    } else {
                        [enc setComputePipelineState:pso_flash_attn_q8_0];
                        [enc setBuffer:d_Q offset:0 atIndex:0];
                        [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                        [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        [enc setBuffer:d_O_attn offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
                        [enc setBytes:&attn_scale length:sizeof(float) atIndex:6];
                        [enc setThreadgroupMemoryLength:16384 atIndex:0];
                        MTLSize grid_fa_q8 = MTLSizeMake((M + 31) / 32, H, 1);
                        [enc dispatchThreadgroups:grid_fa_q8 threadsPerThreadgroup:tg_size_32];
                    }
                });

                // Stage C: O-Projection & First Residual Add
                double t_oproj = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:pso_pipe_gemm_32x32];
                    [enc setBuffer:d_O_attn offset:0 atIndex:0];
                    [enc setBuffer:d_W_o offset:0 atIndex:1];
                    [enc setBuffer:d_O_proj offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:5];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    MTLSize grid_o = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                    [enc dispatchThreadgroups:grid_o threadsPerThreadgroup:tg_size_32];

                    uint32_t num_f4 = (uint32_t)(in_elements / 4);
                    [enc setComputePipelineState:pso_residual_add];
                    [enc setBuffer:d_X_in offset:0 atIndex:0];
                    [enc setBuffer:d_O_proj offset:0 atIndex:1];
                    [enc setBuffer:d_X_mid offset:0 atIndex:2];
                    [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });

                // Stage D & E: MLP Block (Fused Gate+Up SwiGLU + Down GEMM + Residual)
                double t_mlp = time_encoder(^(id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:pso_fused_gate_up];
                    [enc setBuffer:d_X_mid offset:0 atIndex:0];
                    [enc setBuffer:d_W_gate offset:0 atIndex:1];
                    [enc setBuffer:d_W_up offset:0 atIndex:2];
                    [enc setBuffer:d_S_mlp offset:0 atIndex:3];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    MTLSize grid_mlp_up = MTLSizeMake((N_MLP + 31) / 32, (M + 31) / 32, 1);
                    [enc dispatchThreadgroups:grid_mlp_up threadsPerThreadgroup:tg_size_32];

                    [enc setComputePipelineState:pso_pipe_gemm_32x32];
                    [enc setBuffer:d_S_mlp offset:0 atIndex:0];
                    [enc setBuffer:d_W_down offset:0 atIndex:1];
                    [enc setBuffer:d_D_mlp offset:0 atIndex:2];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];
                    MTLSize grid_down = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
                    [enc dispatchThreadgroups:grid_down threadsPerThreadgroup:tg_size_32];

                    uint32_t num_f4 = (uint32_t)(in_elements / 4);
                    [enc setComputePipelineState:pso_residual_add];
                    [enc setBuffer:d_X_mid offset:0 atIndex:0];
                    [enc setBuffer:d_D_mlp offset:0 atIndex:1];
                    [enc setBuffer:d_X_out offset:0 atIndex:2];
                    [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });

                prof.qkv_ms = t_qkv;
                prof.attn_ms = t_attn;
                prof.o_proj_ms = t_oproj;
                prof.mlp_ms = t_mlp;
                prof.total_ms = t_qkv + t_attn + t_oproj + t_mlp;
                prof.throughput_tok_s = ((double)M / (prof.total_ms * 1e-3));
                prof.full_model_32l_ms = prof.total_ms * NUM_LAYERS;
                prof.full_model_throughput_tok_s = ((double)M / (prof.full_model_32l_ms * 1e-3));
                prof.dram_bandwidth_gb_s = ((double)total_weight_bytes_layer / (prof.total_ms * 1e-3)) / 1e9;
            };

            // Warmup runs
            LayerProfile dummy;
            for (int w = 0; w < WARMUP_ITERS; w++) {
                run_baseline_pass(dummy);
                run_unified_pass(false, dummy);
                run_unified_pass(true, dummy);
            }

            // Benchmark runs
            LayerProfile prof_base = {}, prof_opt_fp16 = {}, prof_opt_q8 = {};
            for (int it = 0; it < BENCH_ITERS; it++) {
                LayerProfile p;
                run_baseline_pass(p);
                prof_base.qkv_ms    += p.qkv_ms;
                prof_base.attn_ms   += p.attn_ms;
                prof_base.o_proj_ms += p.o_proj_ms;
                prof_base.mlp_ms    += p.mlp_ms;
                prof_base.total_ms  += p.total_ms;

                run_unified_pass(false, p);
                prof_opt_fp16.qkv_ms    += p.qkv_ms;
                prof_opt_fp16.attn_ms   += p.attn_ms;
                prof_opt_fp16.o_proj_ms += p.o_proj_ms;
                prof_opt_fp16.mlp_ms    += p.mlp_ms;
                prof_opt_fp16.total_ms  += p.total_ms;

                run_unified_pass(true, p);
                prof_opt_q8.qkv_ms    += p.qkv_ms;
                prof_opt_q8.attn_ms   += p.attn_ms;
                prof_opt_q8.o_proj_ms += p.o_proj_ms;
                prof_opt_q8.mlp_ms    += p.mlp_ms;
                prof_opt_q8.total_ms  += p.total_ms;
            }

            auto avg_prof = [&](LayerProfile& p) {
                p.qkv_ms /= BENCH_ITERS;
                p.attn_ms /= BENCH_ITERS;
                p.o_proj_ms /= BENCH_ITERS;
                p.mlp_ms /= BENCH_ITERS;
                p.total_ms /= BENCH_ITERS;
                p.throughput_tok_s = ((double)M / (p.total_ms * 1e-3));
                p.full_model_32l_ms = p.total_ms * NUM_LAYERS;
                p.full_model_throughput_tok_s = ((double)M / (p.full_model_32l_ms * 1e-3));
                p.dram_bandwidth_gb_s = ((double)total_weight_bytes_layer / (p.total_ms * 1e-3)) / 1e9;
            };

            avg_prof(prof_base);
            avg_prof(prof_opt_fp16);
            avg_prof(prof_opt_q8);

            // Print Detailed Breakdown
            std::cout << std::left << std::setw(28) << "Pipeline Stage"
                      << std::right << std::setw(16) << "Baseline (ms)"
                      << std::setw(20) << "Unified FP16 (ms)"
                      << std::setw(20) << "Unified Q8_0 (ms)"
                      << std::setw(16) << "Speedup (FP16)"
                      << std::setw(16) << "Speedup (Q8_0)" << std::endl;
            std::cout << std::string(116, '-') << std::endl;

            auto print_row = [&](const std::string& name, double b, double u_fp16, double u_q8) {
                std::cout << std::left << std::setw(28) << name
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(16) << b
                          << std::setw(20) << u_fp16
                          << std::setw(20) << u_q8
                          << std::setw(15) << (b / u_fp16) << "x"
                          << std::setw(15) << (b / u_q8) << "x" << std::endl;
            };

            print_row("1. QKV Projections", prof_base.qkv_ms, prof_opt_fp16.qkv_ms, prof_opt_q8.qkv_ms);
            print_row("2. Causal Attention Engine", prof_base.attn_ms, prof_opt_fp16.attn_ms, prof_opt_q8.attn_ms);
            print_row("3. Output Proj & Residual", prof_base.o_proj_ms, prof_opt_fp16.o_proj_ms, prof_opt_q8.o_proj_ms);
            print_row("4. MLP Stage (SwiGLU+Down)", prof_base.mlp_ms, prof_opt_fp16.mlp_ms, prof_opt_q8.mlp_ms);
            std::cout << std::string(116, '-') << std::endl;
            print_row("TOTAL 1-LAYER LATENCY", prof_base.total_ms, prof_opt_fp16.total_ms, prof_opt_q8.total_ms);
            print_row("EXTRAPOLATED 32L MODEL", prof_base.full_model_32l_ms, prof_opt_fp16.full_model_32l_ms, prof_opt_q8.full_model_32l_ms);

            std::cout << "\n    [*] 1-Layer Throughput:   Baseline: " << std::fixed << std::setprecision(0) << prof_base.throughput_tok_s
                      << " tok/s | Opt FP16: " << prof_opt_fp16.throughput_tok_s << " tok/s (" << std::setprecision(2) << (prof_opt_fp16.throughput_tok_s / prof_base.throughput_tok_s) << "x)"
                      << " | Opt Q8_0: " << prof_opt_q8.throughput_tok_s << " tok/s (" << (prof_opt_q8.throughput_tok_s / prof_base.throughput_tok_s) << "x)" << std::endl;
            std::cout << "    [*] Full 32L Prefill Time: Baseline: " << std::fixed << std::setprecision(2) << prof_base.full_model_32l_ms << " ms"
                      << " | Opt FP16: " << prof_opt_fp16.full_model_32l_ms << " ms"
                      << " | Opt Q8_0: " << prof_opt_q8.full_model_32l_ms << " ms" << std::endl;
            std::cout << "    [*] DRAM Bandwidth Saturation: " << std::fixed << std::setprecision(1) << prof_opt_fp16.dram_bandwidth_gb_s << " GB/s"
                      << " (Apple M4 LPDDR5X Theoretical Peak: 120.0 GB/s, Utilization: " << std::setprecision(1) << (prof_opt_fp16.dram_bandwidth_gb_s / 120.0 * 100.0) << "%)" << std::endl;

            summary_table.push_back({
                M,
                prof_base.total_ms,
                prof_opt_fp16.total_ms,
                prof_opt_q8.total_ms,
                prof_base.total_ms / prof_opt_fp16.total_ms,
                prof_base.total_ms / prof_opt_q8.total_ms,
                prof_opt_fp16.throughput_tok_s,
                prof_opt_q8.throughput_tok_s,
                prof_opt_fp16.full_model_32l_ms,
                prof_opt_q8.full_model_32l_ms,
                prof_opt_fp16.full_model_throughput_tok_s,
                prof_opt_fp16.dram_bandwidth_gb_s
            });
        }

        // ====================================================================
        // EXECUTIVE SUMMARY TABLE
        // ====================================================================
        std::cout << "\n=================================================================================================================" << std::endl;
        std::cout << "        EXECUTIVE SUMMARY: UNIFIED 8B PREFILL ENGINE vs LLAMA.CPP-STYLE BASELINE ON APPLE M4 (16GB)              \n";
        std::cout << "        Baseline: llama.cpp-style baseline (in-house Metal reimplementation of ggml mul_mm, calibrated ~8-10 TFLOPS on M4)\n";
        std::cout << "=================================================================================================================" << std::endl;
        std::cout << std::left << std::setw(10) << "Prompt M"
                  << std::right << std::setw(14) << "Base (ms/l)"
                  << std::setw(15) << "Opt FP16 (ms)"
                  << std::setw(15) << "Opt Q8_0 (ms)"
                  << std::setw(14) << "Speedup FP16"
                  << std::setw(14) << "Speedup Q8"
                  << std::setw(16) << "1L Tput FP16"
                  << std::setw(16) << "32L Tput FP16"
                  << std::setw(15) << "Full-Model Est"
                  << std::setw(14) << "DRAM BW" << std::endl;
        std::cout << std::string(143, '=') << std::endl;

        for (const auto& row : summary_table) {
            std::cout << std::left << std::setw(10) << row.M
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(14) << row.baseline_ms
                      << std::setw(15) << row.unified_fp16_ms
                      << std::setw(15) << row.unified_q8_ms
                      << std::setw(13) << row.speedup_fp16 << "x"
                      << std::setw(13) << row.speedup_q8 << "x"
                      << std::setw(14) << std::setprecision(0) << row.tok_s_unified_fp16 << " t/s"
                      << std::setw(14) << std::setprecision(0) << row.full_32l_tok_s << " t/s"
                      << std::setw(12) << std::setprecision(2) << row.full_32l_fp16_ms << " ms"
                      << std::setw(11) << std::setprecision(1) << row.dram_bw_gb_s << " GB/s" << std::endl;
        }

        std::cout << "=================================================================================================================" << std::endl;
        std::cout << " [✓] ALL 8B TRANSFORMER SCALE OBJECTIVES AND BENCHMARKS COMPLETED SUCCESSFULLY." << std::endl;
        std::cout << "=================================================================================================================" << std::endl;
    }
    return 0;
}
