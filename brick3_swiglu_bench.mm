#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <sstream>

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct block_q4_0 {
    __fp16 d;
    uint8_t qs[16];
};

static uint32_t prng_state = 2026;
static inline float rand_uniform() {
    prng_state = prng_state * 1664525u + 1013904223u;
    return (float)prng_state / (float)0xFFFFFFFF;
}

void generate_activations(__fp16* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = (__fp16)(rand_uniform() * 2.0f - 1.0f);
    }
}

void generate_q4_0_weights(block_q4_0* blocks, size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = (__fp16)(rand_uniform() * 0.04f + 0.002f);
        for (int i = 0; i < 16; i++) {
            uint8_t low = (uint8_t)(rand_uniform() * 16.0f);
            uint8_t high = (uint8_t)(rand_uniform() * 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

// ============================================================================
// CPU DOUBLE-PRECISION GOLD REFERENCE FOR SWIGLU
// ============================================================================
void cpu_gold_reference_swiglu(
    const __fp16* A,
    const block_q4_0* B_gate,
    const block_q4_0* B_up,
    __fp16* Out,
    uint32_t M,
    uint32_t N_mlp,
    uint32_t K)
{
    uint32_t nb = K / 32;
    for (uint32_t m = 0; m < M; m++) {
        const __fp16* a_row = A + (size_t)m * K;
        for (uint32_t n = 0; n < N_mlp; n++) {
            const block_q4_0* bg_col = B_gate + (size_t)n * nb;
            const block_q4_0* bu_col = B_up + (size_t)n * nb;

            double acc_g = 0.0;
            double acc_u = 0.0;

            for (uint32_t b = 0; b < nb; b++) {
                double dg = (double)bg_col[b].d;
                double du = (double)bu_col[b].d;
                uint32_t a_offset = b * 32;

                for (int i = 0; i < 16; i++) {
                    uint8_t byte_g = bg_col[b].qs[i];
                    int vg0 = (int)(byte_g & 0x0F) - 8;
                    int vg1 = (int)(byte_g >> 4) - 8;

                    uint8_t byte_u = bu_col[b].qs[i];
                    int vu0 = (int)(byte_u & 0x0F) - 8;
                    int vu1 = (int)(byte_u >> 4) - 8;

                    double a0 = (double)a_row[a_offset + i];
                    double a1 = (double)a_row[a_offset + i + 16];

                    acc_g += a0 * ((double)vg0 * dg);
                    acc_g += a1 * ((double)vg1 * dg);

                    acc_u += a0 * ((double)vu0 * du);
                    acc_u += a1 * ((double)vu1 * du);
                }
            }

            // SwiGLU Activation: SiLU(acc_g) * acc_u
            double silu_g = acc_g / (1.0 + std::exp(-acc_g));
            double swiglu = silu_g * acc_u;
            Out[(size_t)m * N_mlp + n] = (__fp16)swiglu;
        }
    }
}

// ============================================================================
// BENCHMARK CONFIGURATION & STRUCTURES
// ============================================================================
enum class KernelVariant {
    ScalarBaseline,
    SplitMlxStyle,
    DualSimdCooperative,
    Cooperative64x64
};

struct VariantConfig {
    KernelVariant variant;
    std::string name;
    std::string description;
};

struct ScaleConfig {
    std::string name;
    uint32_t K;
    uint32_t N_mlp;
};

struct RunMetrics {
    double median_gpu_ms;
    double min_gpu_ms;
    double max_gpu_ms;
    double mean_gpu_ms;
    double host_wall_ms;
    double tflops;
    double pct_mma_peak;
    double effective_gbps;
    double dram_traffic_mb;
    double traffic_saved_mb;
    double traffic_reduction_pct;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "====================================================================================================" << std::endl;
        std::cout << "          J.A.R.V.I.S. BRICK 3 EMPIRICAL MICRO-EXPERIMENT: COOPERATIVE SWIGLU ENGINE BENCHMARK      " << std::endl;
        std::cout << "          Apple M4 (10-Core GPU, 16GB UMA) | Scalar Baseline vs Split MLX vs Dual-SIMD MMA          " << std::endl;
        std::cout << "====================================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[FATAL] Metal is not supported on this host device." << std::endl;
            return 1;
        }

        std::cout << "[+] Active Hardware Device: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"brick3_swiglu_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[FATAL] Failed to read brick3_swiglu_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[FATAL] Metal shader compilation failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // 1. HARDWARE CEILING PROBES
        // --------------------------------------------------------------------
        std::cout << "\n>>> [1] PROBING HARDWARE CEILINGS (MEMORY BANDWIDTH & COMPUTE ROOFLINES)" << std::endl;

        id<MTLFunction> bwFunc = [library newFunctionWithName:@"probe_memory_bandwidth"];
        id<MTLComputePipelineState> bwPipeline = [device newComputePipelineStateWithFunction:bwFunc error:&error];
        size_t bw_elements = 32 * 1024 * 1024;
        size_t bw_bytes = bw_elements * sizeof(float) * 4;
        id<MTLBuffer> bufSrc = [device newBufferWithLength:bw_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [device newBufferWithLength:bw_bytes options:MTLResourceStorageModeShared];

        double peak_gbps = 0.0;
        for (int iter = 0; iter < 15; iter++) {
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:bwPipeline];
            [encoder setBuffer:bufSrc offset:0 atIndex:0];
            [encoder setBuffer:bufDst offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(bw_elements, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];

            __block CFTimeInterval gpuStart = 0;
            __block CFTimeInterval gpuEnd = 0;
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                gpuStart = buffer.GPUStartTime;
                gpuEnd = buffer.GPUEndTime;
            }];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            double elapsed_sec = gpuEnd - gpuStart;
            double gbps = (2.0 * (double)bw_bytes / 1e9) / elapsed_sec;
            if (gbps > peak_gbps) peak_gbps = gbps;
        }
        std::cout << "    [+] Empirical Memory Bandwidth: " << std::fixed << std::setprecision(2) << peak_gbps << " GB/s" << std::endl;

        id<MTLFunction> fmaFunc = [library newFunctionWithName:@"probe_fma_roofline"];
        id<MTLComputePipelineState> fmaPipeline = [device newComputePipelineStateWithFunction:fmaFunc error:&error];
        id<MTLBuffer> fmaOut = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        uint32_t fma_threads = 1024 * 1024;
        double peak_alu_tflops = 0.0;
        for (int iter = 0; iter < 15; iter++) {
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:fmaPipeline];
            [encoder setBuffer:fmaOut offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(fma_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];

            __block CFTimeInterval gpuStart = 0;
            __block CFTimeInterval gpuEnd = 0;
            [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                gpuStart = buffer.GPUStartTime;
                gpuEnd = buffer.GPUEndTime;
            }];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            double elapsed_sec = gpuEnd - gpuStart;
            double total_flops = (double)fma_threads * 6144.0;
            double tflops = (total_flops / 1e12) / elapsed_sec;
            if (tflops > peak_alu_tflops) peak_alu_tflops = tflops;
        }
        const double m4_mma_peak_tflops = 16.8; // Apple M4 10-Core GPU Hardware MMA FP16 Peak
        std::cout << "    [+] Empirical Vector ALU FP16 Roofline: " << std::fixed << std::setprecision(2) << peak_alu_tflops << " TFLOPS" << std::endl;
        std::cout << "    [+] Theoretical Hardware MMA FP16 Peak: " << std::fixed << std::setprecision(2) << m4_mma_peak_tflops << " TFLOPS" << std::endl;

        // --------------------------------------------------------------------
        // 2. PIPELINES COMPILATION
        // --------------------------------------------------------------------
        id<MTLFunction> fn_scalar = [library newFunctionWithName:@"swiglu_scalar_baseline"];
        id<MTLFunction> fn_split_gemm = [library newFunctionWithName:@"gemm_split_q4_0_pass"];
        id<MTLFunction> fn_split_vec4 = [library newFunctionWithName:@"swiglu_split_elementwise_vec4_pass"];
        id<MTLFunction> fn_dual_simd = [library newFunctionWithName:@"swiglu_mma_dual_simd"];
        id<MTLFunction> fn_coop_64x64 = [library newFunctionWithName:@"swiglu_mma_cooperative_64x64"];

        if (!fn_scalar || !fn_split_gemm || !fn_split_vec4 || !fn_dual_simd || !fn_coop_64x64) {
            std::cerr << "[FATAL] Failed to find one or more Metal kernel entrypoints." << std::endl;
            return 1;
        }

        id<MTLComputePipelineState> pso_scalar = [device newComputePipelineStateWithFunction:fn_scalar error:&error];
        id<MTLComputePipelineState> pso_split_gemm = [device newComputePipelineStateWithFunction:fn_split_gemm error:&error];
        id<MTLComputePipelineState> pso_split_vec4 = [device newComputePipelineStateWithFunction:fn_split_vec4 error:&error];
        id<MTLComputePipelineState> pso_dual_simd = [device newComputePipelineStateWithFunction:fn_dual_simd error:&error];
        id<MTLComputePipelineState> pso_coop_64x64 = [device newComputePipelineStateWithFunction:fn_coop_64x64 error:&error];

        std::vector<VariantConfig> variants = {
            {KernelVariant::ScalarBaseline,     "Scalar Baseline (Fused Gate/Up ALU)", "Scalar Vector ALU (64 accumulators/thread)"},
            {KernelVariant::SplitMlxStyle,      "Split Multi-Pass (MLX-Style)",        "Split Gate GEMM -> Up GEMM -> Elementwise"},
            {KernelVariant::DualSimdCooperative,"Dual-SIMD Cooperative (Brick 3)",     "128-thread Dual-SIMDgroup SwiGLU Engine (64x32 Tile)"},
            {KernelVariant::Cooperative64x64,   "Cooperative 64x64 (Octa-SIMD MMA)",   "256-thread Fused SwiGLU Engine (64x64 Tile)"}
        };

        // --------------------------------------------------------------------
        // 3. STRICT NUMERICAL VERIFICATION ACROSS TEST SHAPES
        // --------------------------------------------------------------------
        std::cout << "\n>>> [2] NUMERICAL VERIFICATION AGAINST CPU DOUBLE-PRECISION GOLD REFERENCE" << std::endl;
        std::vector<uint32_t> verify_Ms = {33, 127, 128, 129};
        uint32_t verify_K = 2048;
        uint32_t verify_N = 5632;
        uint32_t verify_nb = verify_K / 32;
        size_t verify_total_blocks = (size_t)verify_N * verify_nb;

        id<MTLBuffer> bufBg_ver = [device newBufferWithLength:verify_total_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBu_ver = [device newBufferWithLength:verify_total_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        generate_q4_0_weights((block_q4_0*)[bufBg_ver contents], verify_total_blocks);
        generate_q4_0_weights((block_q4_0*)[bufBu_ver contents], verify_total_blocks);

        for (uint32_t test_M : verify_Ms) {
            std::cout << "    [*] Verifying M = " << std::setw(4) << test_M << " (K=" << verify_K << ", N_mlp=" << verify_N << ")... ";
            std::cout.flush();

            size_t act_bytes = (size_t)test_M * verify_K * sizeof(__fp16);
            size_t out_bytes = (size_t)test_M * verify_N * sizeof(__fp16);

            id<MTLBuffer> bufA_ver = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
            generate_activations((__fp16*)[bufA_ver contents], (size_t)test_M * verify_K);

            std::vector<__fp16> cpu_ref(test_M * verify_N);
            cpu_gold_reference_swiglu(
                (const __fp16*)[bufA_ver contents],
                (const block_q4_0*)[bufBg_ver contents],
                (const block_q4_0*)[bufBu_ver contents],
                cpu_ref.data(),
                test_M, verify_N, verify_K);

            for (const auto& var : variants) {
                id<MTLBuffer> bufOut_ver = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
                memset([bufOut_ver contents], 0, out_bytes);

                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];

                if (var.variant == KernelVariant::ScalarBaseline) {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso_scalar];
                    [enc setBuffer:bufA_ver offset:0 atIndex:0];
                    [enc setBuffer:bufBg_ver offset:0 atIndex:1];
                    [enc setBuffer:bufBu_ver offset:0 atIndex:2];
                    [enc setBuffer:bufOut_ver offset:0 atIndex:3];
                    [enc setBytes:&test_M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&verify_N length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&verify_K length:sizeof(uint32_t) atIndex:6];
                    [enc setThreadgroupMemoryLength:4096 atIndex:0];

                    NSUInteger tg_x = (verify_N + 31) / 32;
                    NSUInteger tg_y = (test_M + 31) / 32;
                    [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding];
                }
                else if (var.variant == KernelVariant::SplitMlxStyle) {
                    id<MTLBuffer> bufGate_interm = [device newBufferWithLength:out_bytes options:MTLResourceStorageModePrivate];
                    id<MTLBuffer> bufUp_interm   = [device newBufferWithLength:out_bytes options:MTLResourceStorageModePrivate];

                    // Pass 1: Gate GEMM
                    id<MTLComputeCommandEncoder> enc1 = [cmdBuf computeCommandEncoder];
                    [enc1 setComputePipelineState:pso_split_gemm];
                    [enc1 setBuffer:bufA_ver offset:0 atIndex:0];
                    [enc1 setBuffer:bufBg_ver offset:0 atIndex:1];
                    [enc1 setBuffer:bufGate_interm offset:0 atIndex:2];
                    [enc1 setBytes:&test_M length:sizeof(uint32_t) atIndex:3];
                    [enc1 setBytes:&verify_N length:sizeof(uint32_t) atIndex:4];
                    [enc1 setBytes:&verify_K length:sizeof(uint32_t) atIndex:5];
                    [enc1 setThreadgroupMemoryLength:16384 atIndex:0];
                    [enc1 dispatchThreadgroups:MTLSizeMake((verify_N + 63) / 64, (test_M + 63) / 64, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc1 endEncoding];

                    // Pass 2: Up GEMM
                    id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
                    [enc2 setComputePipelineState:pso_split_gemm];
                    [enc2 setBuffer:bufA_ver offset:0 atIndex:0];
                    [enc2 setBuffer:bufBu_ver offset:0 atIndex:1];
                    [enc2 setBuffer:bufUp_interm offset:0 atIndex:2];
                    [enc2 setBytes:&test_M length:sizeof(uint32_t) atIndex:3];
                    [enc2 setBytes:&verify_N length:sizeof(uint32_t) atIndex:4];
                    [enc2 setBytes:&verify_K length:sizeof(uint32_t) atIndex:5];
                    [enc2 setThreadgroupMemoryLength:16384 atIndex:0];
                    [enc2 dispatchThreadgroups:MTLSizeMake((verify_N + 63) / 64, (test_M + 63) / 64, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc2 endEncoding];

                    // Pass 3: Elementwise SwiGLU
                    id<MTLComputeCommandEncoder> enc3 = [cmdBuf computeCommandEncoder];
                    [enc3 setComputePipelineState:pso_split_vec4];
                    [enc3 setBuffer:bufGate_interm offset:0 atIndex:0];
                    [enc3 setBuffer:bufUp_interm offset:0 atIndex:1];
                    [enc3 setBuffer:bufOut_ver offset:0 atIndex:2];
                    uint32_t total_vec4 = (test_M * verify_N + 3) / 4;
                    [enc3 setBytes:&total_vec4 length:sizeof(uint32_t) atIndex:3];
                    [enc3 dispatchThreads:MTLSizeMake(total_vec4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc3 endEncoding];
                }
                else if (var.variant == KernelVariant::DualSimdCooperative) {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso_dual_simd];
                    [enc setBuffer:bufA_ver offset:0 atIndex:0];
                    [enc setBuffer:bufBg_ver offset:0 atIndex:1];
                    [enc setBuffer:bufBu_ver offset:0 atIndex:2];
                    [enc setBuffer:bufOut_ver offset:0 atIndex:3];
                    [enc setBytes:&test_M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&verify_N length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&verify_K length:sizeof(uint32_t) atIndex:6];
                    [enc setThreadgroupMemoryLength:17408 atIndex:0];

                    NSUInteger tg_x = (verify_N + 31) / 32;
                    NSUInteger tg_y = (test_M + 63) / 64;
                    [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }
                else if (var.variant == KernelVariant::Cooperative64x64) {
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso_coop_64x64];
                    [enc setBuffer:bufA_ver offset:0 atIndex:0];
                    [enc setBuffer:bufBg_ver offset:0 atIndex:1];
                    [enc setBuffer:bufBu_ver offset:0 atIndex:2];
                    [enc setBuffer:bufOut_ver offset:0 atIndex:3];
                    [enc setBytes:&test_M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&verify_N length:sizeof(uint32_t) atIndex:5];
                    [enc setBytes:&verify_K length:sizeof(uint32_t) atIndex:6];
                    [enc setThreadgroupMemoryLength:32768 atIndex:0];

                    NSUInteger tg_x = (verify_N + 63) / 64;
                    NSUInteger tg_y = (test_M + 63) / 64;
                    [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                const __fp16* out_ptr = (const __fp16*)[bufOut_ver contents];
                float max_diff = 0.0f;
                for (size_t i = 0; i < (size_t)test_M * verify_N; i++) {
                    float va = (float)out_ptr[i];
                    float vb = (float)cpu_ref[i];
                    if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                        fprintf(stderr, "\n[FATAL] NaN/Inf detected! Variant: %s | Index: %zu | GPU: %f | CPU: %f\n",
                                var.name.c_str(), i, va, vb);
                        assert(false && "Hard abort on NaN/Inf");
                        exit(1);
                    }
                    float diff = std::abs(va - vb);
                    if (diff > max_diff) max_diff = diff;
                }

                std::cout << "\n        [" << var.name << "] MaxDiff = " << max_diff;
                if (max_diff > 0.05f) {
                    std::cerr << " -> [FAIL > 0.05]";
                } else {
                    std::cout << " -> [PASS]";
                }
            }
            std::cout << "\n    --> Finished M=" << test_M << std::endl;
        }

        // --------------------------------------------------------------------
        // 4. BENCHMARK SWEEP (1B and 8B Scales across M)
        // --------------------------------------------------------------------
        std::vector<ScaleConfig> scales = {
            {"1B MLP Shape (Llama-3.2-1B: K=2048, N_mlp=5632)",   2048, 5632},
            {"8B MLP Shape (Llama-3-8B:    K=4096, N_mlp=14336)",  4096, 14336}
        };

        const std::vector<uint32_t> prompt_lengths = {33, 127, 128, 129, 512, 1024, 2048};
        const int WARMUP_ITERS = 10;
        const int MEASURED_ITERS = 20;

        std::cout << "\n>>> [3] HEAD-TO-HEAD BENCHMARK EXECUTION (10 Warmup + 20 Measured Iterations)" << std::endl;

        for (const auto& scale : scales) {
            uint32_t K = scale.K;
            uint32_t N_mlp = scale.N_mlp;

            std::cout << "\n" << std::string(132, '#') << std::endl;
            std::cout << " MODEL SCALE: " << scale.name << " (K = " << K << ", N_mlp = " << N_mlp << ")" << std::endl;
            std::cout << std::string(132, '#') << std::endl;

            uint32_t nb = K / 32;
            size_t total_blocks = (size_t)N_mlp * nb;
            size_t weight_bytes = total_blocks * sizeof(block_q4_0);

            id<MTLBuffer> bufBg = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufBu = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
            generate_q4_0_weights((block_q4_0*)[bufBg contents], total_blocks);
            generate_q4_0_weights((block_q4_0*)[bufBu contents], total_blocks);

            for (uint32_t M : prompt_lengths) {
                std::cout << "\n" << std::string(132, '=') << std::endl;
                std::cout << " PROMPT LENGTH: M = " << std::setw(4) << M
                          << " tokens | Gate/Up Projections: [" << M << ", " << K << "] x [" << K << ", " << N_mlp << "]" << std::endl;
                std::cout << std::string(132, '=') << std::endl;

                std::cout << std::left << std::setw(42) << "Kernel Candidate"
                          << std::setw(12) << "Median (ms)"
                          << std::setw(18) << "Min / Max (ms)"
                          << std::setw(14) << "Host Wall(ms)"
                          << std::setw(16) << "Achieved TFLOPS"
                          << std::setw(12) << "% MMA Peak"
                          << std::setw(15) << "DRAM Traffic"
                          << std::setw(15) << "Traffic Saved"
                          << std::endl;
                std::cout << std::string(132, '-') << std::endl;

                size_t act_bytes = (size_t)M * K * sizeof(__fp16);
                size_t out_bytes = (size_t)M * N_mlp * sizeof(__fp16);

                id<MTLBuffer> bufA = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufOut = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
                generate_activations((__fp16*)[bufA contents], (size_t)M * K);

                // Intermediate buffers for Split variant
                id<MTLBuffer> bufGate_interm = [device newBufferWithLength:out_bytes options:MTLResourceStorageModePrivate];
                id<MTLBuffer> bufUp_interm   = [device newBufferWithLength:out_bytes options:MTLResourceStorageModePrivate];

                // Arithmetic FLOPs for SwiGLU:
                // 2 GEMMs: Gate (2*M*K*N_mlp) + Up (2*M*K*N_mlp) + SwiGLU elementwise (4*M*N_mlp)
                double total_flops = 4.0 * (double)M * (double)N_mlp * (double)K + 4.0 * (double)M * (double)N_mlp;

                // Memory Traffic Models (Bytes):
                // Fused Engine: Read A (2*M*K) + Read Bg (0.5625*K*N) + Read Bu (0.5625*K*N) + Write Out (2*M*N)
                double fused_bytes = (double)act_bytes + 2.0 * (double)weight_bytes + (double)out_bytes;
                // Split Engine: Read A*2 + Read Bg + Read Bu + Write Gate + Write Up + Read Gate + Read Up + Write Out
                double split_bytes = 2.0 * (double)act_bytes + 2.0 * (double)weight_bytes + 5.0 * (double)out_bytes;

                for (const auto& var : variants) {
                    auto execute_iteration = [&](double& host_ms) -> double {
                        auto host_start = std::chrono::high_resolution_clock::now();

                        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];

                        if (var.variant == KernelVariant::ScalarBaseline) {
                            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                            [enc setComputePipelineState:pso_scalar];
                            [enc setBuffer:bufA offset:0 atIndex:0];
                            [enc setBuffer:bufBg offset:0 atIndex:1];
                            [enc setBuffer:bufBu offset:0 atIndex:2];
                            [enc setBuffer:bufOut offset:0 atIndex:3];
                            [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                            [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
                            [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                            [enc setThreadgroupMemoryLength:4096 atIndex:0];

                            NSUInteger tg_x = (N_mlp + 31) / 32;
                            NSUInteger tg_y = (M + 31) / 32;
                            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            [enc endEncoding];
                        }
                        else if (var.variant == KernelVariant::SplitMlxStyle) {
                            // Pass 1: Gate GEMM
                            id<MTLComputeCommandEncoder> enc1 = [cmdBuf computeCommandEncoder];
                            [enc1 setComputePipelineState:pso_split_gemm];
                            [enc1 setBuffer:bufA offset:0 atIndex:0];
                            [enc1 setBuffer:bufBg offset:0 atIndex:1];
                            [enc1 setBuffer:bufGate_interm offset:0 atIndex:2];
                            [enc1 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                            [enc1 setBytes:&N_mlp length:sizeof(uint32_t) atIndex:4];
                            [enc1 setBytes:&K length:sizeof(uint32_t) atIndex:5];
                            [enc1 setThreadgroupMemoryLength:16384 atIndex:0];
                            [enc1 dispatchThreadgroups:MTLSizeMake((N_mlp + 63) / 64, (M + 63) / 64, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            [enc1 endEncoding];

                            // Pass 2: Up GEMM
                            id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
                            [enc2 setComputePipelineState:pso_split_gemm];
                            [enc2 setBuffer:bufA offset:0 atIndex:0];
                            [enc2 setBuffer:bufBu offset:0 atIndex:1];
                            [enc2 setBuffer:bufUp_interm offset:0 atIndex:2];
                            [enc2 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                            [enc2 setBytes:&N_mlp length:sizeof(uint32_t) atIndex:4];
                            [enc2 setBytes:&K length:sizeof(uint32_t) atIndex:5];
                            [enc2 setThreadgroupMemoryLength:16384 atIndex:0];
                            [enc2 dispatchThreadgroups:MTLSizeMake((N_mlp + 63) / 64, (M + 63) / 64, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            [enc2 endEncoding];

                            // Pass 3: Elementwise SwiGLU
                            id<MTLComputeCommandEncoder> enc3 = [cmdBuf computeCommandEncoder];
                            [enc3 setComputePipelineState:pso_split_vec4];
                            [enc3 setBuffer:bufGate_interm offset:0 atIndex:0];
                            [enc3 setBuffer:bufUp_interm offset:0 atIndex:1];
                            [enc3 setBuffer:bufOut offset:0 atIndex:2];
                            uint32_t total_vec4 = (M * N_mlp + 3) / 4;
                            [enc3 setBytes:&total_vec4 length:sizeof(uint32_t) atIndex:3];
                            [enc3 dispatchThreads:MTLSizeMake(total_vec4, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc3 endEncoding];
                        }
                        else if (var.variant == KernelVariant::DualSimdCooperative) {
                            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                            [enc setComputePipelineState:pso_dual_simd];
                            [enc setBuffer:bufA offset:0 atIndex:0];
                            [enc setBuffer:bufBg offset:0 atIndex:1];
                            [enc setBuffer:bufBu offset:0 atIndex:2];
                            [enc setBuffer:bufOut offset:0 atIndex:3];
                            [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                            [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
                            [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                            [enc setThreadgroupMemoryLength:17408 atIndex:0];

                            NSUInteger tg_x = (N_mlp + 31) / 32;
                            NSUInteger tg_y = (M + 63) / 64;
                            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            [enc endEncoding];
                        }
                        else if (var.variant == KernelVariant::Cooperative64x64) {
                            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                            [enc setComputePipelineState:pso_coop_64x64];
                            [enc setBuffer:bufA offset:0 atIndex:0];
                            [enc setBuffer:bufBg offset:0 atIndex:1];
                            [enc setBuffer:bufBu offset:0 atIndex:2];
                            [enc setBuffer:bufOut offset:0 atIndex:3];
                            [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                            [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
                            [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                            [enc setThreadgroupMemoryLength:32768 atIndex:0];

                            NSUInteger tg_x = (N_mlp + 63) / 64;
                            NSUInteger tg_y = (M + 63) / 64;
                            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];

                        auto host_end = std::chrono::high_resolution_clock::now();
                        host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();

                        return (gpuEnd - gpuStart) * 1000.0;
                    };

                    // Warmup
                    for (int i = 0; i < WARMUP_ITERS; i++) {
                        double dummy_host = 0.0;
                        execute_iteration(dummy_host);
                    }

                    // Measurement
                    std::vector<double> gpu_times;
                    std::vector<double> host_times;
                    gpu_times.reserve(MEASURED_ITERS);
                    host_times.reserve(MEASURED_ITERS);

                    for (int i = 0; i < MEASURED_ITERS; i++) {
                        double host_ms = 0.0;
                        double gpu_ms = execute_iteration(host_ms);
                        gpu_times.push_back(gpu_ms);
                        host_times.push_back(host_ms);
                    }

                    std::sort(gpu_times.begin(), gpu_times.end());
                    double median_gpu = gpu_times[MEASURED_ITERS / 2];
                    double min_gpu = gpu_times.front();
                    double max_gpu = gpu_times.back();
                    double mean_host = std::accumulate(host_times.begin(), host_times.end(), 0.0) / host_times.size();

                    double tflops = (total_flops / 1e12) / (median_gpu / 1000.0);
                    double pct_peak = (tflops / m4_mma_peak_tflops) * 100.0;

                    double traffic_bytes = (var.variant == KernelVariant::SplitMlxStyle) ? split_bytes : fused_bytes;
                    double traffic_mb = traffic_bytes / (1024.0 * 1024.0);
                    double traffic_saved_mb = (var.variant == KernelVariant::SplitMlxStyle) ? 0.0 : (split_bytes - fused_bytes) / (1024.0 * 1024.0);
                    std::string saved_str = (var.variant == KernelVariant::SplitMlxStyle) ? "0.0 MB (0%)" :
                        (std::to_string((int)traffic_saved_mb) + " MB (-" + std::to_string((int)((split_bytes - fused_bytes)/split_bytes*100.0)) + "%)");

                    std::stringstream min_max_ss;
                    min_max_ss << "[" << std::fixed << std::setprecision(3) << min_gpu << ", " << max_gpu << "]";

                    std::cout << std::left << std::setw(42) << var.name
                              << std::fixed << std::setprecision(4)
                              << std::setw(12) << median_gpu
                              << std::setw(18) << min_max_ss.str()
                              << std::setw(14) << mean_host
                              << std::setprecision(2)
                              << std::setw(16) << tflops
                              << std::setw(12) << (std::to_string(pct_peak).substr(0, 5) + "%")
                              << std::setw(15) << (std::to_string(traffic_mb).substr(0, 6) + " MB")
                              << std::setw(15) << saved_str
                              << std::endl;
                }
            }
        }

        std::cout << "\n====================================================================================================" << std::endl;
        std::cout << ">>> BRICK 3 BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << "====================================================================================================" << std::endl;
    }
    return 0;
}
