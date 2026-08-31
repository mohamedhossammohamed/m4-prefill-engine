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

// ============================================================================
// DATA STRUCTURES & QUANTIZATION TYPES (Q4_0 & Q8_0)
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

static inline float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================================
// CPU REFERENCE PREFILL LAYER (GOLD STANDARD)
// ============================================================================
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
    float attn_scale)
{
    uint32_t attn_dim = H * D;

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

    // 2. Fused Causal Attention
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
                o_attn_ptr[i * attn_dim + h * D + d] = (__fp16)(acc * inv_sum);
            }
        }
    });

    // 3. O-Proj & Residual
    std::vector<float> O_proj(M * K);
    cpu_gemm_q4_0(O_attn.data(), W_o, O_proj.data(), M, K, attn_dim);

    std::vector<__fp16> X_mid(M * K);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        X_mid[i] = (__fp16)((float)X_in[i] + O_proj[i]);
    }

    // 4. MLP Gate & Up
    std::vector<float> Gate_f(M * N_mlp), Up_f(M * N_mlp);
    cpu_gemm_q4_0(X_mid.data(), W_gate, Gate_f.data(), M, N_mlp, K);
    cpu_gemm_q4_0(X_mid.data(), W_up, Up_f.data(), M, N_mlp, K);

    std::vector<__fp16> S_mlp(M * N_mlp);
    for (size_t i = 0; i < (size_t)M * N_mlp; i++) {
        S_mlp[i] = (__fp16)(silu_f32(Gate_f[i]) * Up_f[i]);
    }

    // 5. MLP Down & Final Residual
    std::vector<float> Down_f(M * K);
    cpu_gemm_q4_0(S_mlp.data(), W_down, Down_f.data(), M, K, N_mlp);

    for (size_t i = 0; i < (size_t)M * K; i++) {
        X_out[i] = (__fp16)((float)X_mid[i] + Down_f[i]);
    }
}

// ============================================================================
// TELEMETRY SAMPLING STRUCTURES
// ============================================================================
struct IterationTelemetry {
    double timestamp_sec; // Seconds from start
    double latency_ms;    // Single layer latency
};

struct PeriodicSample {
    double timestamp_sec;
    uint32_t cumulative_iters;
    double window_latency_ms;
    double window_tok_s;
    double window_tflops;
    double max_diff_vs_gold;
    std::string thermal_state_str;
};

// Convert NSProcessInfoThermalState to string
std::string get_thermal_state_string() {
    NSProcessInfoThermalState state = [[NSProcessInfo processInfo] thermalState];
    switch (state) {
        case NSProcessInfoThermalStateNominal:  return "Nominal";
        case NSProcessInfoThermalStateFair:     return "Fair";
        case NSProcessInfoThermalStateSerious:  return "Serious";
        case NSProcessInfoThermalStateCritical: return "Critical";
        default:                                return "Unknown";
    }
}

// ============================================================================
// MAIN SUSTAINED THERMAL STRESS ENGINE
// ============================================================================
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "  J.A.R.V.I.S. SUSTAINED PASSIVE THERMAL STRESS & ROOFLINE SOAK ENGINE (BRICK 14)       " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[-] Error: Metal device initialization failed." << std::endl;
            return 1;
        }

        std::cout << "[+] Target SoC Hardware:      " << [[device name] UTF8String] << std::endl;
        std::cout << "[+] Form Factor Profile:     Fanless Ultra-Thin Passive Chassis (Apple M4)" << std::endl;
        std::cout << "[+] Initial Thermal State:   " << get_thermal_state_string() << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"unified_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[-] Error loading unified_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[-] Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // 1B TRANSFORMER ARCHITECTURE DEFINITION
        // --------------------------------------------------------------------
        const uint32_t K = 2048;         // Hidden Dimension
        const uint32_t H = 16;           // Attention Heads
        const uint32_t D = 64;           // Head Dimension
        const uint32_t ATTN_DIM = H * D; // 1024
        const uint32_t N_MLP = 5632;     // Intermediate MLP Dimension
        const float attn_scale = 1.0f / std::sqrt((float)D); // 0.125f
        const uint32_t M = 2048;         // Full Context Prefill Tokens (2048)

        // Mathematical Floating Point Operation (FLOP) count per prefill layer pass:
        // 1. QKV Projections: 3 * (2 * M * K * ATTN_DIM) = 25,769,803,776 FLOPs
        // 2. Causal FlashAttention: 2 * (2 * H * (M*(M+1)/2) * D) = 8,594,128,896 FLOPs
        // 3. Attention O-Projection: 2 * M * ATTN_DIM * K = 8,589,934,592 FLOPs
        // 4. MLP Gate + Up + Down Projections: 3 * (2 * M * K * N_MLP) = 141,733,920,768 FLOPs
        // Total Layer FLOPs = 184,687,788,032 FLOPs = 0.184687788 TFLOPs
        const double FLOPS_PER_LAYER = 2.0 * (double)M * (double)K * (3.0 * (double)ATTN_DIM)
                                     + 4.0 * (double)H * (double)D * ((double)M * ((double)M + 1.0) / 2.0)
                                     + 2.0 * (double)M * (double)ATTN_DIM * (double)K
                                     + 6.0 * (double)M * (double)K * (double)N_MLP;
        const double TFLOPS_PER_LAYER = FLOPS_PER_LAYER / 1e12;

        std::cout << "\n[+] 1B Transformer Prefill Workload Parameters:" << std::endl;
        std::cout << "    - Sequence Context Length (M):   " << M << " tokens" << std::endl;
        std::cout << "    - Hidden Dimension (K):          " << K << std::endl;
        std::cout << "    - Attention Heads (H):           " << H << " (D = " << D << ", AttnDim = " << ATTN_DIM << ")" << std::endl;
        std::cout << "    - SwiGLU MLP Dimension (N_mlp):  " << N_MLP << std::endl;
        std::cout << "    - Arithmetic Intensity per Pass: " << std::fixed << std::setprecision(3)
                  << (FLOPS_PER_LAYER / 1e9) << " GFLOPs (" << TFLOPS_PER_LAYER << " TFLOPs/layer)" << std::endl;

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

        id<MTLComputePipelineState> pso_pipe_qkv_head   = load_pso(@"pipe_qkv_head_gemm_q4_0");
        id<MTLComputePipelineState> pso_flash_attn_fp16 = load_pso(@"flash_attn_fp16_causal");
        id<MTLComputePipelineState> pso_pipe_gemm_32x32 = load_pso(@"pipe_gemm_q4_0_32x32");
        id<MTLComputePipelineState> pso_residual_add    = load_pso(@"vector_add_residual");
        id<MTLComputePipelineState> pso_fused_gate_up   = load_pso(@"fused_gate_up_swiglu_q4_0");

        // --------------------------------------------------------------------
        // ALLOCATE WEIGHTS & ACTIVATIONS
        // --------------------------------------------------------------------
        size_t qkv_blocks      = (size_t)ATTN_DIM * (K / 32);
        size_t o_blocks        = (size_t)K * (ATTN_DIM / 32);
        size_t mlp_up_blocks   = (size_t)N_MLP * (K / 32);
        size_t mlp_down_blocks = (size_t)K * (N_MLP / 32);

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

        size_t in_elements = (size_t)M * K;
        std::vector<__fp16> h_X_in(in_elements);
        generate_activations(h_X_in.data(), in_elements);

        id<MTLBuffer> d_X_in   = [device newBufferWithBytes:h_X_in.data() length:in_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_Q      = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_K      = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_V      = [device newBufferWithLength:(size_t)H * M * D * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_O_attn = [device newBufferWithLength:(size_t)M * ATTN_DIM * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_O_proj = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_X_mid  = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_S_mlp  = [device newBufferWithLength:(size_t)M * N_MLP * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_D_mlp  = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_X_out  = [device newBufferWithLength:(size_t)M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];

        // --------------------------------------------------------------------
        // ENCODER HELPER: DISPATCH COMPLETE UNIFIED PREFILL LAYER
        // --------------------------------------------------------------------
        MTLSize grid_qkv    = MTLSizeMake((ATTN_DIM + 31) / 32, (M + 31) / 32, 1);
        MTLSize grid_fa     = MTLSizeMake((M + 31) / 32, H, 1);
        MTLSize grid_o      = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
        MTLSize grid_mlp_up = MTLSizeMake((N_MLP + 31) / 32, (M + 31) / 32, 1);
        MTLSize grid_down   = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
        MTLSize tg_size_32  = MTLSizeMake(32, 1, 1);
        uint32_t num_f4     = (uint32_t)(in_elements / 4);

        auto encode_full_prefill_layer = [&](id<MTLComputeCommandEncoder> enc) {
            // Stage 1: Q, K, V Direct Head Projections
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

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 2: Fused FlashAttention FP16 Causal
            [enc setComputePipelineState:pso_flash_attn_fp16];
            [enc setBuffer:d_Q offset:0 atIndex:0];
            [enc setBuffer:d_K offset:0 atIndex:1];
            [enc setBuffer:d_V offset:0 atIndex:2];
            [enc setBuffer:d_O_attn offset:0 atIndex:3];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&attn_scale length:sizeof(float) atIndex:6];
            [enc setThreadgroupMemoryLength:8192 atIndex:0];
            [enc dispatchThreadgroups:grid_fa threadsPerThreadgroup:tg_size_32];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 3: Output Projection (O_attn @ W_o -> O_proj)
            [enc setComputePipelineState:pso_pipe_gemm_32x32];
            [enc setBuffer:d_O_attn offset:0 atIndex:0];
            [enc setBuffer:d_W_o offset:0 atIndex:1];
            [enc setBuffer:d_O_proj offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&ATTN_DIM length:sizeof(uint32_t) atIndex:5];
            [enc setThreadgroupMemoryLength:4096 atIndex:0];
            [enc dispatchThreadgroups:grid_o threadsPerThreadgroup:tg_size_32];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 4: Residual Addition (X_mid = X_in + O_proj)
            [enc setComputePipelineState:pso_residual_add];
            [enc setBuffer:d_X_in offset:0 atIndex:0];
            [enc setBuffer:d_O_proj offset:0 atIndex:1];
            [enc setBuffer:d_X_mid offset:0 atIndex:2];
            [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 5: Fused Gate + Up GEMM with SwiGLU Epilogue
            [enc setComputePipelineState:pso_fused_gate_up];
            [enc setBuffer:d_X_mid offset:0 atIndex:0];
            [enc setBuffer:d_W_gate offset:0 atIndex:1];
            [enc setBuffer:d_W_up offset:0 atIndex:2];
            [enc setBuffer:d_S_mlp offset:0 atIndex:3];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
            [enc setThreadgroupMemoryLength:4096 atIndex:0];
            [enc dispatchThreadgroups:grid_mlp_up threadsPerThreadgroup:tg_size_32];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 6: Down Projection (S_mlp @ W_down -> D_mlp)
            [enc setComputePipelineState:pso_pipe_gemm_32x32];
            [enc setBuffer:d_S_mlp offset:0 atIndex:0];
            [enc setBuffer:d_W_down offset:0 atIndex:1];
            [enc setBuffer:d_D_mlp offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&N_MLP length:sizeof(uint32_t) atIndex:5];
            [enc setThreadgroupMemoryLength:4096 atIndex:0];
            [enc dispatchThreadgroups:grid_down threadsPerThreadgroup:tg_size_32];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Stage 7: Final Residual Addition (X_out = X_mid + D_mlp)
            [enc setComputePipelineState:pso_residual_add];
            [enc setBuffer:d_X_mid offset:0 atIndex:0];
            [enc setBuffer:d_D_mlp offset:0 atIndex:1];
            [enc setBuffer:d_X_out offset:0 atIndex:2];
            [enc setBytes:&num_f4 length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(num_f4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        // --------------------------------------------------------------------
        // NUMERICAL GOLD REFERENCE & COLD VERIFICATION
        // --------------------------------------------------------------------
        std::cout << "\n>>> [STAGE 1] ESTABLISHING GOLD REFERENCE & COLD NUMERICAL INTEGRITY..." << std::endl;
        std::vector<__fp16> h_X_gold_cpu(in_elements);
        std::vector<__fp16> h_X_gold_gpu(in_elements);

        std::cout << "    [+] Generating CPU Reference output for M = " << M << "..." << std::flush;
        auto t_cpu_0 = std::chrono::high_resolution_clock::now();
        cpu_reference_prefill_layer(
            h_X_in.data(), h_W_q.data(), h_W_k.data(), h_W_v.data(), h_W_o.data(),
            h_W_gate.data(), h_W_up.data(), h_W_down.data(), h_X_gold_cpu.data(),
            M, K, H, D, N_MLP, attn_scale);
        auto t_cpu_1 = std::chrono::high_resolution_clock::now();
        std::cout << " Done (" << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t_cpu_1 - t_cpu_0).count() << " ms)" << std::endl;

        // Warmup / Cold GPU pass
        {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_full_prefill_layer(enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            memcpy(h_X_gold_gpu.data(), [d_X_out contents], in_elements * sizeof(__fp16));
        }

        // Validate Cold GPU vs CPU
        float cold_max_diff = 0.0f;
        for (size_t i = 0; i < in_elements; i++) {
            float va = (float)h_X_gold_gpu[i];
            float vb = (float)h_X_gold_cpu[i];
            if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                assert(false && "Numerical instability detected (NaN/Inf)");
                exit(1);
            }
            float diff = std::fabs(va - vb);
            if (diff > cold_max_diff) cold_max_diff = diff;
        }
        std::cout << "    [+] Cold GPU vs CPU Max Absolute Difference: " << cold_max_diff
                  << " (Threshold: <= 0.05) -> [" << (cold_max_diff <= 0.05f ? "PASS" : "FAIL") << "]" << std::endl;
        assert(cold_max_diff <= 0.05f);

        // --------------------------------------------------------------------
        // STAGE 2: 60-SECOND SUSTAINED FULL GPU SATURATION SOAK
        // --------------------------------------------------------------------
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 2] EXECUTING 60-SECOND SUSTAINED FULL GPU SATURATION SOAK TEST" << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "    Continuous execution under full compute + memory controller pressure." << std::endl;
        std::cout << "    Sampling telemetry & verifying numerical stability every 5 seconds...\n" << std::endl;

        const double TEST_DURATION_SEC = 60.0;
        const double SAMPLE_INTERVAL_SEC = 5.0;

        std::vector<IterationTelemetry> all_iterations;
        all_iterations.reserve(10000);

        std::vector<PeriodicSample> periodic_samples;
        periodic_samples.reserve(15);

        // Header for time-series output
        std::cout << std::left << std::setw(10) << "Elapsed(s)"
                  << std::right << std::setw(12) << "Cumulative"
                  << std::setw(16) << "Latency (ms)"
                  << std::setw(18) << "Throughput (t/s)"
                  << std::setw(16) << "TFLOPS"
                  << std::setw(14) << "MaxDiff"
                  << std::setw(18) << "Thermal State" << std::endl;
        std::cout << std::string(104, '-') << std::endl;

        auto soak_start_time = std::chrono::high_resolution_clock::now();
        double next_sample_target = SAMPLE_INTERVAL_SEC;
        size_t last_sample_iter_idx = 0;

        while (true) {
            @autoreleasepool {
                __block CFTimeInterval gpuStart = 0;
                __block CFTimeInterval gpuEnd = 0;

                id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                encode_full_prefill_layer(enc);
                [enc endEncoding];

                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                    gpuStart = buffer.GPUStartTime;
                    gpuEnd = buffer.GPUEndTime;
                }];

                [cmd commit];
                [cmd waitUntilCompleted];

                double iter_latency_ms = (gpuEnd - gpuStart) * 1000.0;
                auto iter_t1 = std::chrono::high_resolution_clock::now();
                double elapsed_soak_sec = std::chrono::duration<double>(iter_t1 - soak_start_time).count();

                all_iterations.push_back({elapsed_soak_sec, iter_latency_ms});

                // Check if 5-second interval boundary reached
                if (elapsed_soak_sec >= next_sample_target || elapsed_soak_sec >= TEST_DURATION_SEC) {
                    size_t iters_in_window = all_iterations.size() - last_sample_iter_idx;
                    double window_latency_sum = 0.0;
                    for (size_t i = last_sample_iter_idx; i < all_iterations.size(); i++) {
                        window_latency_sum += all_iterations[i].latency_ms;
                    }
                    double avg_window_latency_ms = (iters_in_window > 0) ? (window_latency_sum / iters_in_window) : iter_latency_ms;
                    double window_tok_s = (double)M / (avg_window_latency_ms * 1e-3);
                    double window_tflops = TFLOPS_PER_LAYER / (avg_window_latency_ms * 1e-3);

                    // Check live numerical accuracy against CPU gold baseline
                    const __fp16* current_gpu_out = (const __fp16*)[d_X_out contents];
                    float max_diff_sample = 0.0f;
                    for (size_t i = 0; i < in_elements; i++) {
                        float va = (float)current_gpu_out[i];
                        float vb = (float)h_X_gold_cpu[i];
                        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                            fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                            assert(false && "Numerical instability detected (NaN/Inf)");
                            exit(1);
                        }
                        float d = std::fabs(va - vb);
                        if (d > max_diff_sample) max_diff_sample = d;
                    }
                    assert(max_diff_sample <= 0.05f && "Numerical drift detected during heat soak!");

                    std::string therm_str = get_thermal_state_string();

                    PeriodicSample sample = {
                        elapsed_soak_sec,
                        (uint32_t)all_iterations.size(),
                        avg_window_latency_ms,
                        window_tok_s,
                        window_tflops,
                        max_diff_sample,
                        therm_str
                    };
                    periodic_samples.push_back(sample);

                    // Print periodic row
                    std::cout << std::left << std::setw(10) << std::fixed << std::setprecision(1) << elapsed_soak_sec
                              << std::right << std::setw(12) << all_iterations.size()
                              << std::setw(16) << std::fixed << std::setprecision(3) << avg_window_latency_ms
                              << std::setw(18) << std::fixed << std::setprecision(1) << window_tok_s
                              << std::setw(16) << std::fixed << std::setprecision(2) << window_tflops
                              << std::setw(14) << std::scientific << std::setprecision(2) << max_diff_sample
                              << std::setw(18) << therm_str << std::endl;

                    last_sample_iter_idx = all_iterations.size();
                    next_sample_target += SAMPLE_INTERVAL_SEC;
                }

                if (elapsed_soak_sec >= TEST_DURATION_SEC) {
                    break;
                }
            }
        }

        std::cout << std::string(104, '-') << std::endl;
        std::cout << "[✓] 60-SECOND FULL GPU SATURATION SOAK COMPLETED SUCCESSFULLY." << std::endl;

        // --------------------------------------------------------------------
        // STAGE 3: THERMAL TELEMETRY & STABILITY ANALYSIS
        // --------------------------------------------------------------------
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 3] THERMAL TELEMETRY & SYSTEMIC STABILITY DEGRADATION ANALYSIS" << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        // 1. Cold Burst Peak Window: First 3.0 seconds
        std::vector<double> cold_latencies;
        for (const auto& it : all_iterations) {
            if (it.timestamp_sec <= 3.0) {
                cold_latencies.push_back(it.latency_ms);
            }
        }
        if (cold_latencies.empty()) cold_latencies.push_back(all_iterations.front().latency_ms);

        double cold_mean_ms = std::accumulate(cold_latencies.begin(), cold_latencies.end(), 0.0) / cold_latencies.size();
        double cold_min_ms  = *std::min_element(cold_latencies.begin(), cold_latencies.end());
        double cold_max_ms  = *std::max_element(cold_latencies.begin(), cold_latencies.end());
        double cold_tok_s   = (double)M / (cold_mean_ms * 1e-3);
        double cold_tflops  = TFLOPS_PER_LAYER / (cold_mean_ms * 1e-3);
        double cold_peak_tflops = TFLOPS_PER_LAYER / (cold_min_ms * 1e-3);

        // 2. Heat-Soaked Sustained Window: Final 15.0 seconds (45s to 60s)
        std::vector<double> sustained_latencies;
        for (const auto& it : all_iterations) {
            if (it.timestamp_sec >= 45.0) {
                sustained_latencies.push_back(it.latency_ms);
            }
        }
        if (sustained_latencies.empty()) sustained_latencies.push_back(all_iterations.back().latency_ms);

        double sustained_mean_ms = std::accumulate(sustained_latencies.begin(), sustained_latencies.end(), 0.0) / sustained_latencies.size();
        double sustained_min_ms  = *std::min_element(sustained_latencies.begin(), sustained_latencies.end());
        double sustained_max_ms  = *std::max_element(sustained_latencies.begin(), sustained_latencies.end());
        double sustained_tok_s   = (double)M / (sustained_mean_ms * 1e-3);
        double sustained_tflops  = TFLOPS_PER_LAYER / (sustained_mean_ms * 1e-3);

        // 3. Overall Statistics & Coefficient of Variation (CV)
        std::vector<double> all_latencies;
        all_latencies.reserve(all_iterations.size());
        for (const auto& it : all_iterations) all_latencies.push_back(it.latency_ms);

        double total_mean_ms = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / all_latencies.size();
        double variance_sum = 0.0;
        for (double l : all_latencies) {
            variance_sum += (l - total_mean_ms) * (l - total_mean_ms);
        }
        double std_dev_ms = std::sqrt(variance_sum / all_latencies.size());
        double cv_percent = (std_dev_ms / total_mean_ms) * 100.0;

        // Sustained phase CV
        double sustained_var_sum = 0.0;
        for (double l : sustained_latencies) {
            sustained_var_sum += (l - sustained_mean_ms) * (l - sustained_mean_ms);
        }
        double sustained_std_dev_ms = std::sqrt(sustained_var_sum / sustained_latencies.size());
        double sustained_cv_percent = (sustained_std_dev_ms / sustained_mean_ms) * 100.0;

        // 4. Throttling Degradation Delta
        double delta_latency_ms = sustained_mean_ms - cold_mean_ms;
        double latency_inflation_pct = (delta_latency_ms / cold_mean_ms) * 100.0;
        double delta_tflops = cold_tflops - sustained_tflops;
        double throttling_degradation_pct = (delta_tflops / cold_tflops) * 100.0;

        // Total Work Done
        size_t total_iters = all_iterations.size();
        double total_tokens_processed = (double)total_iters * (double)M;
        double total_tflops_work = (double)total_iters * TFLOPS_PER_LAYER;
        double total_petaflops_work = total_tflops_work / 1000.0;

        std::cout << "\n------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "  THERMAL METRIC COMPARISON: COLD BURST vs HEAT-SOAKED SUSTAINED (APPLE M4)" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;
        std::cout << std::left << std::setw(36) << "Thermal Metric"
                  << std::right << std::setw(22) << "Cold Burst (0-3s)"
                  << std::setw(24) << "Heat-Soaked (45-60s)"
                  << std::setw(20) << "Delta / Ratio" << std::endl;
        std::cout << std::string(102, '-') << std::endl;

        std::cout << std::left << std::setw(36) << "Mean 1-Layer Latency"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(19) << cold_mean_ms << " ms"
                  << std::setw(21) << sustained_mean_ms << " ms"
                  << std::setw(19) << std::showpos << delta_latency_ms << " ms (" << latency_inflation_pct << "%)" << std::noshowpos << std::endl;

        std::cout << std::left << std::setw(36) << "Latency Range [Min, Max]"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << "[" << cold_min_ms << ", " << cold_max_ms << "] ms"
                  << std::setw(13) << "[" << sustained_min_ms << ", " << sustained_max_ms << "] ms"
                  << std::setw(19) << "-" << std::endl;

        std::cout << std::left << std::setw(36) << "Instantaneous Throughput"
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(17) << cold_tok_s << " tok/s"
                  << std::setw(19) << sustained_tok_s << " tok/s"
                  << std::setw(19) << std::showpos << (sustained_tok_s - cold_tok_s) << " tok/s" << std::noshowpos << std::endl;

        std::cout << std::left << std::setw(36) << "Effective Compute Roofline"
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(16) << cold_tflops << " TFLOPS"
                  << std::setw(17) << sustained_tflops << " TFLOPS"
                  << std::setw(19) << std::showpos << (-delta_tflops) << " TFLOPS (" << -throttling_degradation_pct << "%)" << std::noshowpos << std::endl;

        std::cout << std::left << std::setw(36) << "Peak Transient Burst Compute"
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(16) << cold_peak_tflops << " TFLOPS"
                  << std::setw(24) << "N/A"
                  << std::setw(20) << "Transient Peak" << std::endl;

        std::cout << std::string(102, '-') << std::endl;

        // Final Post-Soak Verification vs CPU Gold
        const __fp16* final_gpu_out = (const __fp16*)[d_X_out contents];
        float final_max_diff = 0.0f;
        for (size_t i = 0; i < in_elements; i++) {
            float va = (float)final_gpu_out[i];
            float vb = (float)h_X_gold_cpu[i];
            if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                assert(false && "Numerical instability detected (NaN/Inf)");
                exit(1);
            }
            float diff = std::fabs(va - vb);
            if (diff > final_max_diff) final_max_diff = diff;
        }
        assert(final_max_diff <= 0.05f);

        std::cout << "\n------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "  SYSTEMIC STABILITY & CHASSIS DYNAMICS TELEMETRY" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "  - Total Iterations Executed (60s):      " << total_iters << " layer passes" << std::endl;
        std::cout << "  - Total Prefill Tokens Processed:       " << std::fixed << std::setprecision(0) << total_tokens_processed << " tokens (" << (total_tokens_processed / 1e6) << " Mtok)" << std::endl;
        std::cout << "  - Cumulative Arithmetic Work:           " << std::fixed << std::setprecision(3) << total_tflops_work << " TFLOPs (" << total_petaflops_work << " PFLOPs)" << std::endl;
        std::cout << "  - Mean Execution Latency:               " << std::fixed << std::setprecision(3) << total_mean_ms << " ms" << std::endl;
        std::cout << "  - Overall Latency Standard Deviation:   " << std::fixed << std::setprecision(3) << std_dev_ms << " ms" << std::endl;
        std::cout << "  - Overall Stability Score (CV):         " << std::fixed << std::setprecision(2) << cv_percent << " %" << std::endl;
        std::cout << "  - Sustained Phase Stability Score (CV): " << std::fixed << std::setprecision(2) << sustained_cv_percent << " %" << std::endl;
        std::cout << "  - Thermal Throttling Degradation:       " << std::fixed << std::setprecision(2) << throttling_degradation_pct << " %" << std::endl;
        std::cout << "  - Final MaxDiff vs CPU Gold:            " << std::fixed << std::setprecision(5) << final_max_diff << " (Threshold: <= 0.05) -> [PASS]" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;

        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << " [✓] [D5.2] SUSTAINED PASSIVE THERMAL STRESS TEST EXECUTION COMPLETED." << std::endl;
        std::cout << "==========================================================================================" << std::endl;
    }
    return 0;
}
