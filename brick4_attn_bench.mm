#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
// DATA STRUCTURES
// ============================================================================
struct block_q8_0 {
    __fp16 d;
    int8_t qs[32];
};

static uint32_t prng_state = 2026;
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

void quantize_to_q8_0(const __fp16* src, block_q8_0* dst, size_t num_elements) {
    size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; b++) {
        const __fp16* s = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float val = std::fabs((float)s[i]);
            if (val > amax) amax = val;
        }
        float d = amax / 127.0f;
        dst[b].d = (__fp16)d;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        for (int i = 0; i < 32; i++) {
            int q = (int)std::round((float)s[i] * id);
            q = std::max(-128, std::min(127, q));
            dst[b].qs[i] = (int8_t)q;
        }
    }
}

void dequantize_q8_0(const block_q8_0* src, __fp16* dst, size_t num_elements) {
    size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; b++) {
        float d = (float)src[b].d;
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = (__fp16)((float)src[b].qs[i] * d);
        }
    }
}

// ============================================================================
// CPU DOUBLE-PRECISION GOLD REFERENCE FOR CAUSAL ATTENTION
// ============================================================================
void cpu_gold_reference_causal_attention(
    const __fp16* Q, // [H, M, D]
    const __fp16* K, // [H, M, D]
    const __fp16* V, // [H, M, D]
    __fp16* O,       // [H, M, D]
    uint32_t H,
    uint32_t M,
    uint32_t D,
    float scale)
{
    for (uint32_t h = 0; h < H; h++) {
        const __fp16* q_head = Q + (size_t)h * M * D;
        const __fp16* k_head = K + (size_t)h * M * D;
        const __fp16* v_head = V + (size_t)h * M * D;
        __fp16* o_head = O + (size_t)h * M * D;

        for (uint32_t i = 0; i < M; i++) {
            const __fp16* q_row = q_head + (size_t)i * D;
            std::vector<double> scores(i + 1);
            double max_s = -1e30;

            for (uint32_t j = 0; j <= i; j++) {
                const __fp16* k_row = k_head + (size_t)j * D;
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) {
                    dot += (double)q_row[d] * (double)k_row[d];
                }
                double s = dot * (double)scale;
                scores[j] = s;
                if (s > max_s) max_s = s;
            }

            double sum_exp = 0.0;
            for (uint32_t j = 0; j <= i; j++) {
                scores[j] = std::exp(scores[j] - max_s);
                sum_exp += scores[j];
            }

            double inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j <= i; j++) {
                    double p = scores[j] * inv_sum;
                    acc += p * (double)v_head[(size_t)j * D + d];
                }
                o_head[(size_t)i * D + d] = (__fp16)acc;
            }
        }
    }
}

// ============================================================================
// BENCHMARK CONFIGURATION & METRICS
// ============================================================================
enum class KernelType {
    ScalarBaseline,
    MmaFP16,
    MmaQ8_0
};

struct KernelVariant {
    KernelType type;
    std::string name;
    std::string func_d64;
    std::string func_d128;
    bool is_q8_0;
};

struct ScaleConfig {
    std::string name;
    uint32_t H;
    uint32_t D;
};

struct RunMetrics {
    double median_gpu_ms;
    double min_gpu_ms;
    double max_gpu_ms;
    double mean_gpu_ms;
    double host_wall_ms;
    double tflops;
    double kv_footprint_mb;
    double speedup;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "====================================================================================================" << std::endl;
        std::cout << "          J.A.R.V.I.S. BRICK 4 EMPIRICAL MICRO-EXPERIMENT: 2D BLOCK-MMA FLASHATTENTION              " << std::endl;
        std::cout << "          Apple M4 (10-Core GPU, 16GB UMA) | Scalar Baseline vs MMA FP16 vs Dynamic Q8_0 KV Cache  " << std::endl;
        std::cout << "====================================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[FATAL] Metal is not supported on this host device." << std::endl;
            return 1;
        }

        std::cout << "[+] Active Hardware Device: " << [[device name] UTF8String] << std::endl;
        std::cout << "[+] Max Threadgroup Memory: " << [device maxThreadgroupMemoryLength] / 1024 << " KB" << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"brick4_attn_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[FATAL] Failed to read brick4_attn_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[FATAL] Metal shader compilation failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // KERNEL DESCRIPTORS
        // --------------------------------------------------------------------
        std::vector<KernelVariant> variants = {
            {KernelType::ScalarBaseline, "Scalar Baseline (1D Vector ALU, BR=32, BC=16)", "flash_attn_scalar_baseline_d64", "flash_attn_scalar_baseline_d128", false},
            {KernelType::MmaFP16,        "2D BlockMMA FlashAttn (FP16 KV Cache, BQ=64, BK=64)", "flash_attn_mma_64x64_fp16_d64",   "flash_attn_mma_64x64_fp16_d128",   false},
            {KernelType::MmaQ8_0,        "2D BlockMMA FlashAttn (Dynamic Q8_0 KV, BQ=64, BK=64)", "flash_attn_mma_64x64_q8_0_d64",   "flash_attn_mma_64x64_q8_0_d128",   true}
        };

        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        for (const auto& var : variants) {
            for (const std::string& fn_name : {var.func_d64, var.func_d128}) {
                id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:fn_name.c_str()]];
                if (!fn) {
                    std::cerr << "[FATAL] Entry point not found: " << fn_name << std::endl;
                    return 1;
                }
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
                if (error) {
                    std::cerr << "[FATAL] PSO compilation failed for " << fn_name << ": " << [[error localizedDescription] UTF8String] << std::endl;
                    return 1;
                }
                pipelines[fn_name] = pso;
            }
        }

        // --------------------------------------------------------------------
        // 1. STRICT NUMERICAL VERIFICATION AGAINST CPU DOUBLE-PRECISION REFERENCE
        // --------------------------------------------------------------------
        std::cout << "\n>>> [1] STRICT NUMERICAL VERIFICATION AGAINST CPU DOUBLE-PRECISION GOLD REFERENCE (MaxDiff <= 0.05)" << std::endl;
        const uint32_t H_VERIFY = 8;
        const std::vector<uint32_t> verify_Ms = {33, 127, 128, 129, 256};
        const std::vector<uint32_t> verify_Ds = {64, 128};

        for (uint32_t test_D : verify_Ds) {
            float scale = 1.0f / std::sqrt((float)test_D);
            std::cout << "\n    ======================== Verification for D = " << test_D << " (scale = " << scale << ") ========================" << std::endl;

            for (uint32_t test_M : verify_Ms) {
                std::cout << "    [*] Verifying Shape [H=" << H_VERIFY << ", M=" << std::setw(3) << test_M << ", D=" << test_D << "] ... ";
                std::cout.flush();

                size_t total_elements = (size_t)H_VERIFY * test_M * test_D;
                std::vector<__fp16> h_Q(total_elements);
                std::vector<__fp16> h_K(total_elements);
                std::vector<__fp16> h_V(total_elements);
                std::vector<__fp16> h_O_cpu_fp16(total_elements);
                std::vector<__fp16> h_O_cpu_q8(total_elements);

                generate_activations(h_Q.data(), total_elements);
                generate_activations(h_K.data(), total_elements);
                generate_activations(h_V.data(), total_elements);

                // Compute CPU Reference for FP16
                cpu_gold_reference_causal_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu_fp16.data(), H_VERIFY, test_M, test_D, scale);

                // Quantize K and V for Q8_0
                size_t total_blocks = total_elements / 32;
                std::vector<block_q8_0> h_K_q8(total_blocks);
                std::vector<block_q8_0> h_V_q8(total_blocks);
                quantize_to_q8_0(h_K.data(), h_K_q8.data(), total_elements);
                quantize_to_q8_0(h_V.data(), h_V_q8.data(), total_elements);

                // Compute CPU Reference for Q8_0 (dequantized)
                std::vector<__fp16> h_K_dequant(total_elements);
                std::vector<__fp16> h_V_dequant(total_elements);
                dequantize_q8_0(h_K_q8.data(), h_K_dequant.data(), total_elements);
                dequantize_q8_0(h_V_q8.data(), h_V_dequant.data(), total_elements);
                cpu_gold_reference_causal_attention(h_Q.data(), h_K_dequant.data(), h_V_dequant.data(), h_O_cpu_q8.data(), H_VERIFY, test_M, test_D, scale);

                // Allocate GPU Buffers
                id<MTLBuffer> d_Q = [device newBufferWithBytes:h_Q.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_K = [device newBufferWithBytes:h_K.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_V = [device newBufferWithBytes:h_V.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_K_q8 = [device newBufferWithBytes:h_K_q8.data() length:total_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_V_q8 = [device newBufferWithBytes:h_V_q8.data() length:total_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_O = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];

                for (const auto& var : variants) {
                    memset([d_O contents], 0, total_elements * sizeof(__fp16));
                    std::string func_name = (test_D == 64) ? var.func_d64 : var.func_d128;
                    id<MTLComputePipelineState> pso = pipelines[func_name];

                    id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:d_Q offset:0 atIndex:0];

                    if (var.is_q8_0) {
                        [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                        [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                    } else {
                        [enc setBuffer:d_K offset:0 atIndex:1];
                        [enc setBuffer:d_V offset:0 atIndex:2];
                    }

                    [enc setBuffer:d_O offset:0 atIndex:3];
                    [enc setBytes:&test_M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&scale length:sizeof(float) atIndex:5];

                    if (var.type == KernelType::ScalarBaseline) {
                        [enc dispatchThreadgroups:MTLSizeMake((test_M + 31) / 32, H_VERIFY, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        NSUInteger smem_len = (test_D == 64) ? 32768 : 57344;
                        [enc setThreadgroupMemoryLength:smem_len atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake((test_M + 63) / 64, H_VERIFY, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    [enc endEncoding];
                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];

                    const __fp16* gpu_out = (const __fp16*)[d_O contents];
                    const __fp16* cpu_ref = var.is_q8_0 ? h_O_cpu_q8.data() : h_O_cpu_fp16.data();

                    float max_diff = 0.0f;
                    double sum_diff = 0.0;
                    for (size_t i = 0; i < total_elements; i++) {
                        float va = (float)gpu_out[i];
                        float vb = (float)cpu_ref[i];
                        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                            std::cerr << "\n[FATAL] Numerical Instability (NaN/Inf) detected in " << var.name
                                      << " at index " << i << "! GPU: " << va << " | CPU: " << vb << std::endl;
                            assert(false && "Hard abort on NaN/Inf");
                            exit(1);
                        }
                        float diff = std::fabs(va - vb);
                        if (diff > max_diff) max_diff = diff;
                        sum_diff += diff;
                    }
                    float avg_diff = (float)(sum_diff / total_elements);

                    if (max_diff > 0.05f) {
                        std::cerr << "\n[FAIL] " << var.name << " MaxDiff = " << max_diff << " (threshold 0.05)" << std::endl;
                        assert(false && "Verification assertion failed");
                        exit(1);
                    }
                }
                std::cout << "PASS (All 3 variants MaxDiff <= 0.05)" << std::endl;
            }
        }

        // --------------------------------------------------------------------
        // 2. HEAD-TO-HEAD BENCHMARK EXECUTION
        // --------------------------------------------------------------------
        std::cout << "\n>>> [2] HEAD-TO-HEAD BENCHMARK EXECUTION (10 Warmup + 20 Measured Iterations)" << std::endl;

        std::vector<ScaleConfig> scales = {
            {"1B Attention Shape (Llama-3.2-1B: H = 32, D = 64)",  32, 64},
            {"8B Attention Shape (Llama-3-8B:    H = 32, D = 128)", 32, 128}
        };

        const std::vector<uint32_t> seq_lengths = {33, 127, 128, 129, 512, 1024, 2048};
        const int WARMUP_ITERS = 10;
        const int MEASURED_ITERS = 20;

        for (const auto& scale_cfg : scales) {
            uint32_t H = scale_cfg.H;
            uint32_t D = scale_cfg.D;
            float scale = 1.0f / std::sqrt((float)D);

            std::cout << "\n" << std::string(115, '=') << std::endl;
            std::cout << " SCALE: " << scale_cfg.name << " | Scale factor = " << scale << std::endl;
            std::cout << std::string(115, '=') << std::endl;

            for (uint32_t M : seq_lengths) {
                std::cout << "\n>>> Sequence Length M = " << std::setw(4) << M
                          << " | Causal Pairs = " << (uint64_t)M * (M + 1) / 2
                          << " | Total FLOPs = " << std::fixed << std::setprecision(2)
                          << (2.0 * H * M * (M + 1) * D) / 1e9 << " GFLOPs" << std::endl;

                size_t total_elements = (size_t)H * M * D;
                size_t total_blocks = total_elements / 32;

                id<MTLBuffer> d_Q = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_K = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_V = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_K_q8 = [device newBufferWithLength:total_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_V_q8 = [device newBufferWithLength:total_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
                id<MTLBuffer> d_O = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];

                generate_activations((__fp16*)[d_Q contents], total_elements);
                generate_activations((__fp16*)[d_K contents], total_elements);
                generate_activations((__fp16*)[d_V contents], total_elements);
                quantize_to_q8_0((const __fp16*)[d_K contents], (block_q8_0*)[d_K_q8 contents], total_elements);
                quantize_to_q8_0((const __fp16*)[d_V contents], (block_q8_0*)[d_V_q8 contents], total_elements);

                double baseline_latency = 0.0;
                std::map<KernelType, RunMetrics> results;

                for (const auto& var : variants) {
                    std::string func_name = (D == 64) ? var.func_d64 : var.func_d128;
                    id<MTLComputePipelineState> pso = pipelines[func_name];

                    // Warmup Iterations
                    for (int iter = 0; iter < WARMUP_ITERS; iter++) {
                        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:d_Q offset:0 atIndex:0];

                        if (var.is_q8_0) {
                            [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                            [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        } else {
                            [enc setBuffer:d_K offset:0 atIndex:1];
                            [enc setBuffer:d_V offset:0 atIndex:2];
                        }

                        [enc setBuffer:d_O offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&scale length:sizeof(float) atIndex:5];

                        if (var.type == KernelType::ScalarBaseline) {
                            [enc dispatchThreadgroups:MTLSizeMake((M + 31) / 32, H, 1)
                                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        } else {
                            NSUInteger smem_len = (D == 64) ? 32768 : 57344;
                            [enc setThreadgroupMemoryLength:smem_len atIndex:0];
                            [enc dispatchThreadgroups:MTLSizeMake((M + 63) / 64, H, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        [enc endEncoding];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                    }

                    // Measured Iterations
                    std::vector<double> gpu_times;
                    std::vector<double> host_times;

                    for (int iter = 0; iter < MEASURED_ITERS; iter++) {
                        auto host_start = std::chrono::high_resolution_clock::now();

                        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:d_Q offset:0 atIndex:0];

                        if (var.is_q8_0) {
                            [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                            [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        } else {
                            [enc setBuffer:d_K offset:0 atIndex:1];
                            [enc setBuffer:d_V offset:0 atIndex:2];
                        }

                        [enc setBuffer:d_O offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&scale length:sizeof(float) atIndex:5];

                        if (var.type == KernelType::ScalarBaseline) {
                            [enc dispatchThreadgroups:MTLSizeMake((M + 31) / 32, H, 1)
                                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        } else {
                            NSUInteger smem_len = (D == 64) ? 32768 : 57344;
                            [enc setThreadgroupMemoryLength:smem_len atIndex:0];
                            [enc dispatchThreadgroups:MTLSizeMake((M + 63) / 64, H, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        [enc endEncoding];

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];

                        auto host_end = std::chrono::high_resolution_clock::now();
                        double host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
                        double gpu_ms = (gpuEnd - gpuStart) * 1000.0;

                        gpu_times.push_back(gpu_ms);
                        host_times.push_back(host_ms);
                    }

                    std::sort(gpu_times.begin(), gpu_times.end());
                    double median_gpu = gpu_times[MEASURED_ITERS / 2];
                    double min_gpu = gpu_times.front();
                    double max_gpu = gpu_times.back();
                    double mean_gpu = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / MEASURED_ITERS;
                    double median_host = host_times[MEASURED_ITERS / 2];

                    double total_flops = 2.0 * (double)H * (double)M * ((double)M + 1.0) * (double)D;
                    double tflops = (total_flops / 1e12) / (median_gpu / 1000.0);

                    // KV Cache Footprint: FP16 = 4 * H * M * D bytes; Q8_0 = 2 * H * M * (D/32) * 34 bytes
                    double kv_bytes = var.is_q8_0 ? (2.0 * H * M * (D / 32) * 34.0) : (4.0 * H * M * D);
                    double kv_mb = kv_bytes / (1024.0 * 1024.0);

                    if (var.type == KernelType::ScalarBaseline) {
                        baseline_latency = median_gpu;
                    }
                    double speedup = baseline_latency / median_gpu;

                    results[var.type] = {median_gpu, min_gpu, max_gpu, mean_gpu, median_host, tflops, kv_mb, speedup};
                }

                // Print Comparison Table for this M
                std::cout << std::left << std::setw(36) << "Variant"
                          << std::right << std::setw(14) << "GPU Med (ms)"
                          << std::setw(14) << "GPU Min (ms)"
                          << std::setw(14) << "GPU Max (ms)"
                          << std::setw(14) << "TFLOPS"
                          << std::setw(12) << "Speedup"
                          << std::setw(12) << "KV (MB)"
                          << std::endl;
                std::cout << std::string(115, '-') << std::endl;

                for (const auto& var : variants) {
                    const auto& res = results[var.type];
                    std::cout << std::left << std::setw(36) << var.name
                              << std::right << std::fixed << std::setprecision(3)
                              << std::setw(14) << res.median_gpu_ms
                              << std::setw(14) << res.min_gpu_ms
                              << std::setw(14) << res.max_gpu_ms
                              << std::setprecision(2)
                              << std::setw(14) << res.tflops
                              << std::setw(11) << res.speedup << "x"
                              << std::setw(12) << res.kv_footprint_mb
                              << std::endl;
                }
            }
        }

        std::cout << "\n====================================================================================================" << std::endl;
        std::cout << "          BRICK 4 EMPIRICAL BENCHMARK COMPLETE. ALL ASSERTIONS PASSED.                              " << std::endl;
        std::cout << "====================================================================================================" << std::endl;
    }
    return 0;
}
