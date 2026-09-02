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

struct block_q4_0 {
    __fp16 d;
    uint8_t qs[16];
};

static uint32_t prng_state = 1337;
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

// 2D Block-Swizzling Transformation: [N, K/32] -> [N/64, K/32, 64]
void swizzle_q4_0_weights(
    const block_q4_0* linear_B,
    block_q4_0* swizzled_B,
    uint32_t N,
    uint32_t K)
{
    uint32_t num_k_blocks = K / 32;
    uint32_t num_n_tiles = (N + 63) / 64;

    for (uint32_t tn = 0; tn < num_n_tiles; tn++) {
        for (uint32_t tk = 0; tk < num_k_blocks; tk++) {
            for (uint32_t c = 0; c < 64; c++) {
                uint32_t col = tn * 64 + c;
                size_t swizzled_idx = ((size_t)tn * num_k_blocks + tk) * 64 + c;
                if (col < N) {
                    size_t linear_idx = (size_t)col * num_k_blocks + tk;
                    swizzled_B[swizzled_idx] = linear_B[linear_idx];
                } else {
                    memset(&swizzled_B[swizzled_idx], 0, sizeof(block_q4_0));
                }
            }
        }
    }
}

// CPU Gold Reference with Double-Precision Accumulation
void cpu_gold_reference_q4_0(
    const __fp16* A,
    const block_q4_0* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K)
{
    uint32_t nb = K / 32;
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            const block_q4_0* b_col = B + (size_t)n * nb;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;

            for (uint32_t b = 0; b < nb; b++) {
                double d = (double)b_col[b].d;
                uint32_t a_offset = b * 32;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = b_col[b].qs[i];
                    int v0 = (int)(byte_val & 0x0F) - 8;
                    int v1 = (int)(byte_val >> 4) - 8;
                    acc += (double)a_row[a_offset + i] * ((double)v0 * d);
                    acc += (double)a_row[a_offset + i + 16] * ((double)v1 * d);
                }
            }
            C[m * N + n] = (__fp16)acc;
        }
    }
}

struct KernelConfig {
    std::string category;      // "Brick 1 Baseline", "Double Buffered", "Fused 3-in-1"
    std::string name;          // Display name
    std::string function_name; // Metal function symbol
    bool is_swizzled;          // Requires 2D swizzled weights
    uint32_t tg_m;             // Tile M (64)
    uint32_t tg_n;             // Tile N (64)
    uint32_t tg_threads;       // Threads per threadgroup (128)
    uint32_t shmem_bytes;      // Dynamic shared memory bytes
};

struct BenchmarkResult {
    std::string category;
    std::string name;
    double median_gpu_ms;
    double min_gpu_ms;
    double max_gpu_ms;
    double mean_gpu_ms;
    double host_wall_ms;
    double tflops;
    double pct_mma_peak;
    double pct_alu_peak;
    double bandwidth_gbps;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "====================================================================================================" << std::endl;
        std::cout << "          J.A.R.V.I.S. BRICK 2 EMPIRICAL MICRO-EXPERIMENT: FUSED 3-IN-1 PIPELINE BENCHMARK          " << std::endl;
        std::cout << "               Apple M4 (10-Core GPU, 16GB UMA) | Baseline vs Ping-Pong vs Fused 3-in-1             " << std::endl;
        std::cout << "====================================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[FATAL] Metal is not supported on this host device." << std::endl;
            return 1;
        }

        std::cout << "[+] Active Hardware Device: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"brick2_fused_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[FATAL] Failed to read brick2_fused_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
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
        // 2. KERNEL CANDIDATES SETUP
        // --------------------------------------------------------------------
        std::vector<KernelConfig> candidates = {
            {"Baseline",        "gemm_mma_q4_0_64x64_baseline        (Brick 1 Winner)",   "gemm_mma_q4_0_64x64_baseline",        false, 64, 64, 128, 16384},
            {"Double-Buffered", "gemm_mma_q4_0_64x64_double_buffered (Padded SRAM Staging)", "gemm_mma_q4_0_64x64_double_buffered", false, 64, 64, 128, 17408},
            {"Fused Pipeline",  "gemm_mma_q4_0_64x64_fused_pipeline  (Fused 3-in-1)",       "gemm_mma_q4_0_64x64_fused_pipeline",  true,  64, 64, 128, 17408}
        };

        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        for (const auto& kc : candidates) {
            id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:kc.function_name.c_str()]];
            if (!fn) {
                std::cerr << "[FATAL] Missing kernel function: " << kc.function_name << std::endl;
                return 1;
            }
            pipelines[kc.name] = [device newComputePipelineStateWithFunction:fn error:&error];
            if (error) {
                std::cerr << "[FATAL] Pipeline creation error for " << kc.name << ": " << [[error localizedDescription] UTF8String] << std::endl;
                return 1;
            }
        }

        // --------------------------------------------------------------------
        // 3. STRICT NUMERICAL VERIFICATION ACROSS TEST SHAPES
        // --------------------------------------------------------------------
        std::cout << "\n>>> [2] NUMERICAL VERIFICATION AGAINST CPU DOUBLE-PRECISION GOLD REFERENCE" << std::endl;
        std::vector<uint32_t> verify_Ms = {33, 127, 128, 129};
        uint32_t verify_K = 2048;
        uint32_t verify_N = 2048;

        uint32_t verify_nb = verify_K / 32;
        size_t verify_total_blocks = (size_t)verify_N * verify_nb;
        id<MTLBuffer> bufB_linear = [device newBufferWithLength:verify_total_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB_swizzled = [device newBufferWithLength:verify_total_blocks * sizeof(block_q4_0) options:MTLResourceStorageModeShared];

        generate_q4_0_weights((block_q4_0*)[bufB_linear contents], verify_total_blocks);
        swizzle_q4_0_weights((const block_q4_0*)[bufB_linear contents], (block_q4_0*)[bufB_swizzled contents], verify_N, verify_K);

        for (uint32_t test_M : verify_Ms) {
            std::cout << "    [*] Verifying M = " << std::setw(4) << test_M << " (K=" << verify_K << ", N=" << verify_N << ")... ";
            std::cout.flush();

            size_t act_bytes = (size_t)test_M * verify_K * sizeof(__fp16);
            size_t out_bytes = (size_t)test_M * verify_N * sizeof(__fp16);
            id<MTLBuffer> bufA_test = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
            generate_activations((__fp16*)[bufA_test contents], (size_t)test_M * verify_K);

            std::vector<__fp16> cpu_ref(test_M * verify_N);
            cpu_gold_reference_q4_0((const __fp16*)[bufA_test contents], (const block_q4_0*)[bufB_linear contents], cpu_ref.data(), test_M, verify_N, verify_K);

            for (const auto& kc : candidates) {
                id<MTLBuffer> bufC_test = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
                memset([bufC_test contents], 0, out_bytes);

                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pipelines[kc.name]];
                [enc setBuffer:bufA_test offset:0 atIndex:0];
                [enc setBuffer:(kc.is_swizzled ? bufB_swizzled : bufB_linear) offset:0 atIndex:1];
                [enc setBuffer:bufC_test offset:0 atIndex:2];
                [enc setBytes:&test_M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&verify_N length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&verify_K length:sizeof(uint32_t) atIndex:5];

                if (kc.shmem_bytes > 0) {
                    [enc setThreadgroupMemoryLength:kc.shmem_bytes atIndex:0];
                }

                NSUInteger tg_x = (verify_N + kc.tg_n - 1) / kc.tg_n;
                NSUInteger tg_y = (test_M + kc.tg_m - 1) / kc.tg_m;
                [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(kc.tg_threads, 1, 1)];
                [enc endEncoding];

                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                const __fp16* out_ptr = (const __fp16*)[bufC_test contents];
                float max_diff = 0.0f;
                for (size_t i = 0; i < (size_t)test_M * verify_N; i++) {
                    float va = (float)out_ptr[i];
                    float vb = (float)cpu_ref[i];
                    if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                        fprintf(stderr, "\n[FATAL] NaN/Inf detected! Kernel: %s | Index: %zu | GPU: %f | CPU: %f\n", kc.name.c_str(), i, va, vb);
                        assert(false && "Hard abort on NaN/Inf");
                        exit(1);
                    }
                    float diff = std::abs(va - vb);
                    if (diff > max_diff) max_diff = diff;
                }

                if (max_diff > 0.05f) {
                    std::cerr << "\n[FAIL] Assertion MaxDiff <= 0.05 failed! Kernel: " << kc.name 
                              << " M=" << test_M << " MaxDiff=" << max_diff << std::endl;
                    assert(false && "MaxDiff assertion failed!");
                    exit(1);
                }
            }
            std::cout << "All 3 kernels PASSED (MaxDiff <= 0.05, 0 NaN/Inf)" << std::endl;
        }

        // --------------------------------------------------------------------
        // 4. BENCHMARK SWEEP (1B and 8B Scales across M)
        // --------------------------------------------------------------------
        struct ScaleConfig {
            std::string name;
            uint32_t K;
            uint32_t N;
        };

        std::vector<ScaleConfig> scales = {
            {"1B Projection Shape", 2048, 2048},
            {"8B Projection Shape", 4096, 4096}
        };

        const std::vector<uint32_t> prompt_lengths = {33, 127, 128, 129, 512, 1024, 2048};
        const int WARMUP_ITERS = 10;
        const int MEASURED_ITERS = 20;

        std::cout << "\n>>> [3] HEAD-TO-HEAD BENCHMARK EXECUTION (10 Warmup + 20 Measured Iterations)" << std::endl;

        for (const auto& scale : scales) {
            uint32_t K = scale.K;
            uint32_t N = scale.N;

            std::cout << "\n" << std::string(125, '#') << std::endl;
            std::cout << " MODEL SCALE: " << scale.name << " (K = " << K << ", N = " << N << ")" << std::endl;
            std::cout << std::string(125, '#') << std::endl;

            uint32_t nb = K / 32;
            size_t total_blocks = (size_t)N * nb;
            size_t weight_bytes = total_blocks * sizeof(block_q4_0);
            id<MTLBuffer> bufB_linear = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufB_swizzled = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];

            generate_q4_0_weights((block_q4_0*)[bufB_linear contents], total_blocks);
            swizzle_q4_0_weights((const block_q4_0*)[bufB_linear contents], (block_q4_0*)[bufB_swizzled contents], N, K);

            for (uint32_t M : prompt_lengths) {
                std::cout << "\n" << std::string(125, '=') << std::endl;
                std::cout << " PROMPT LENGTH: M = " << std::setw(4) << M << " tokens | Shape: [" << M << ", " << K << "] x [" << K << ", " << N << "]" << std::endl;
                std::cout << std::string(125, '=') << std::endl;

                std::cout << std::left << std::setw(50) << "Kernel Candidate"
                          << std::setw(12) << "Median (ms)"
                          << std::setw(18) << "Min / Max (ms)"
                          << std::setw(14) << "Host Wall(ms)"
                          << std::setw(14) << "Achieved TFLOPS"
                          << std::setw(12) << "% MMA Peak"
                          << std::setw(15) << "Bandwidth(GB/s)"
                          << std::endl;
                std::cout << std::string(125, '-') << std::endl;

                size_t act_bytes = (size_t)M * K * sizeof(__fp16);
                size_t out_bytes = (size_t)M * N * sizeof(__fp16);
                id<MTLBuffer> bufA = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufC = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
                generate_activations((__fp16*)[bufA contents], (size_t)M * K);

                for (const auto& kc : candidates) {
                    id<MTLBuffer> target_bufB = kc.is_swizzled ? bufB_swizzled : bufB_linear;

                    auto execute_iteration = [&](double& host_ms) -> double {
                        auto host_start = std::chrono::high_resolution_clock::now();

                        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                        [enc setComputePipelineState:pipelines[kc.name]];
                        [enc setBuffer:bufA offset:0 atIndex:0];
                        [enc setBuffer:target_bufB offset:0 atIndex:1];
                        [enc setBuffer:bufC offset:0 atIndex:2];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];

                        if (kc.shmem_bytes > 0) {
                            [enc setThreadgroupMemoryLength:kc.shmem_bytes atIndex:0];
                        }

                        NSUInteger tg_x = (N + kc.tg_n - 1) / kc.tg_n;
                        NSUInteger tg_y = (M + kc.tg_m - 1) / kc.tg_m;
                        [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(kc.tg_threads, 1, 1)];
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
                        host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();

                        return (gpuEnd - gpuStart) * 1000.0;
                    };

                    // Warmup
                    double dummy_host = 0.0;
                    for (int w = 0; w < WARMUP_ITERS; w++) {
                        execute_iteration(dummy_host);
                    }

                    // Measured sampling
                    std::vector<double> gpu_times(MEASURED_ITERS);
                    std::vector<double> host_times(MEASURED_ITERS);
                    for (int i = 0; i < MEASURED_ITERS; i++) {
                        gpu_times[i] = execute_iteration(host_times[i]);
                    }

                    std::sort(gpu_times.begin(), gpu_times.end());
                    std::sort(host_times.begin(), host_times.end());

                    double median_gpu = gpu_times[MEASURED_ITERS / 2];
                    double min_gpu = gpu_times.front();
                    double max_gpu = gpu_times.back();
                    double median_host = host_times[MEASURED_ITERS / 2];

                    double total_flops = 2.0 * (double)M * (double)N * (double)K;
                    double tflops = (total_flops / 1e12) / (median_gpu / 1000.0);
                    double pct_mma = (tflops / m4_mma_peak_tflops) * 100.0;

                    // Theoretical minimum memory traffic: A read + B read + C write
                    double total_bytes = (double)act_bytes + (double)weight_bytes + (double)out_bytes;
                    double bw_gbps = (total_bytes / 1e9) / (median_gpu / 1000.0);

                    std::string min_max_str = "[" + std::to_string(min_gpu).substr(0, 5) + ", " + std::to_string(max_gpu).substr(0, 5) + "]";

                    std::cout << std::left << std::setw(50) << kc.name
                              << std::setw(12) << std::fixed << std::setprecision(4) << median_gpu
                              << std::setw(18) << min_max_str
                              << std::setw(14) << std::fixed << std::setprecision(4) << median_host
                              << std::setw(14) << std::fixed << std::setprecision(2) << tflops
                              << std::setw(12) << std::fixed << std::setprecision(2) << (std::to_string(pct_mma).substr(0, 5) + "%")
                              << std::setw(15) << std::fixed << std::setprecision(2) << bw_gbps
                              << std::endl;
                }
            }
        }

        std::cout << "\n" << std::string(125, '=') << std::endl;
        std::cout << ">>> BRICK 2 BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << std::string(125, '=') << std::endl;
    }
    return 0;
}
