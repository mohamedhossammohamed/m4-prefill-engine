#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <map>

struct block_q4_0 {
    __fp16 d;
    uint8_t qs[16];
};

static uint32_t prng_state = 42;
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
        blocks[b].d = (__fp16)(rand_uniform() * 0.05f + 0.001f);
        for (int i = 0; i < 16; i++) {
            uint8_t low = (uint8_t)(rand_uniform() * 16.0f);
            uint8_t high = (uint8_t)(rand_uniform() * 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

void cpu_reference_gemm(
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
            const block_q4_0* b_col = B + n * nb;
            const __fp16* a_row = A + m * K;
            float acc = 0.0f;

            for (uint32_t b = 0; b < nb; b++) {
                float d = (float)b_col[b].d;
                uint32_t a_offset = b * 32;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = b_col[b].qs[i];
                    int v0 = (int)(byte_val & 0x0F) - 8;
                    int v1 = (int)(byte_val >> 4) - 8;
                    acc += (float)a_row[a_offset + i] * ((float)v0 * d);
                    acc += (float)a_row[a_offset + i + 16] * ((float)v1 * d);
                }
            }
            C[m * N + n] = (__fp16)acc;
        }
    }
}

struct KernelConfig {
    std::string name;
    std::string function_name;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tg_m;
    uint32_t tg_n;
    uint32_t tg_threads;
    uint32_t shmem_a_bytes;
    uint32_t shmem_b_bytes;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "      J.A.R.V.I.S. M4 EMPIRICAL MICRO-BENCHMARK: 4-WAY FUSED ALU UNPACKING ENGINE        " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return 1;
        }

        std::cout << "[+] Hardware: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"micro_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "Error loading micro_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // 1. HARDWARE PROBING
        std::cout << "\n>>> [1] PROBING HARDWARE CEILINGS (PEAK BANDWIDTH & FP16 ROOFLINE)" << std::endl;
        
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
        double peak_tflops = 0.0;
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
            if (tflops > peak_tflops) peak_tflops = tflops;
        }
        std::cout << "    [+] Empirical FP16 Compute Roofline: " << std::fixed << std::setprecision(2) << peak_tflops << " TFLOPS" << std::endl;

        // 2. KERNEL MATRIX
        std::vector<KernelConfig> kernels = {
            {"llama.cpp (Drawer 8x4)", "llamacpp_style_mul_mm_q4_0", 8, 4, 64, 32, 64, 4096, 2048},
            {"Fused-Col (Tile 16x1)",  "fused_col_16x1",             16, 1, 16, 64, 64, 1024, 0},
            {"Fused-Col (Tile 16x2)",  "fused_col_16x2",             16, 2, 16, 128, 64, 1024, 0},
            {"Fused-Col (Tile 32x1)",  "fused_col_32x1",             32, 1, 32, 64, 64, 2048, 0},
            {"Fused-Col (Tile 32x2)",  "fused_col_32x2",             32, 2, 32, 128, 64, 2048, 0},
            {"Fused-Col (Tile 64x1)",  "fused_col_64x1",             64, 1, 64, 64, 64, 4096, 0},
            {"Fused-2D  (Tile 4x4)",   "fused_q4_gemm_4x4",          4, 4, 32, 32, 64, 2048, 0},
            {"Fused-2D  (Tile 8x4)",   "fused_q4_gemm_8x4",          8, 4, 64, 32, 64, 4096, 0},
            {"Fused-2D  (Tile 8x8)",   "fused_q4_gemm_8x8",          8, 8, 64, 64, 64, 4096, 0},
            {"Fused-2D  (Tile 4x8)",   "fused_q4_gemm_4x8",          4, 8, 32, 64, 64, 2048, 0},
            {"Fused-2D  (Tile 16x4)",  "fused_q4_gemm_16x4",         16, 4, 128, 32, 64, 8192, 0}
        };

        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        for (const auto& kc : kernels) {
            id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:kc.function_name.c_str()]];
            if (!fn) {
                std::cerr << "Failed to find kernel function: " << kc.function_name << std::endl;
                return 1;
            }
            pipelines[kc.name] = [device newComputePipelineStateWithFunction:fn error:&error];
            if (error) {
                std::cerr << "Pipeline creation failed for: " << kc.name << " : " << [[error localizedDescription] UTF8String] << std::endl;
                return 1;
            }
        }

        const uint32_t K = 2048;
        const uint32_t N = 2048;
        const std::vector<uint32_t> prompt_lengths = {128, 512, 1024, 2048};

        uint32_t blocks_per_col = K / 32;
        size_t total_blocks = (size_t)N * blocks_per_col;
        size_t weight_bytes = total_blocks * sizeof(block_q4_0);
        id<MTLBuffer> bufB = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        generate_q4_0_weights((block_q4_0*)[bufB contents], total_blocks);

        // 3. CORRECTNESS VERIFICATION
        std::cout << "\n>>> [2] NUMERICAL CORRECTNESS VERIFICATION (K=2048, N=2048, M=128)" << std::endl;
        uint32_t verify_M = 128;
        size_t verify_act_bytes = (size_t)verify_M * K * sizeof(__fp16);
        size_t verify_out_bytes = (size_t)verify_M * N * sizeof(__fp16);
        id<MTLBuffer> bufA_verify = [device newBufferWithLength:verify_act_bytes options:MTLResourceStorageModeShared];
        generate_activations((__fp16*)[bufA_verify contents], (size_t)verify_M * K);

        std::vector<__fp16> cpu_ref(verify_M * N);
        std::cout << "    [*] Computing CPU gold standard reference (M=128)... ";
        std::cout.flush();
        cpu_reference_gemm((const __fp16*)[bufA_verify contents], (const block_q4_0*)[bufB contents], cpu_ref.data(), verify_M, N, K);
        std::cout << "Done." << std::endl;

        for (const auto& kc : kernels) {
            id<MTLBuffer> bufC = [device newBufferWithLength:verify_out_bytes options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipelines[kc.name]];
            [encoder setBuffer:bufA_verify offset:0 atIndex:0];
            [encoder setBuffer:bufB offset:0 atIndex:1];
            [encoder setBuffer:bufC offset:0 atIndex:2];
            [encoder setBytes:&verify_M length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

            if (kc.shmem_a_bytes > 0) [encoder setThreadgroupMemoryLength:kc.shmem_a_bytes atIndex:0];
            if (kc.shmem_b_bytes > 0) [encoder setThreadgroupMemoryLength:kc.shmem_b_bytes atIndex:1];

            NSUInteger tg_x = (N + kc.tg_n - 1) / kc.tg_n;
            NSUInteger tg_y = (verify_M + kc.tg_m - 1) / kc.tg_m;
            [encoder dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(kc.tg_threads, 1, 1)];
            [encoder endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            const __fp16* out_ptr = (const __fp16*)[bufC contents];
            float max_diff = 0.0f;
            for (size_t i = 0; i < (size_t)verify_M * N; i++) {
                float va = (float)out_ptr[i];
                float vb = (float)cpu_ref[i];
                if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                    fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                    assert(false && "Numerical instability detected (NaN/Inf)");
                    exit(1);
                }
                float d = std::abs(va - vb);
                if (d > max_diff) max_diff = d;
            }

            bool passed = (max_diff <= 0.05f);
            std::cout << "    [" << (passed ? "PASS" : "FAIL") << "] " << std::left << std::setw(26) << kc.name 
                      << " MaxDiff vs CPU = " << std::fixed << std::setprecision(5) << max_diff << std::endl;
            assert(passed && "Numerical correctness verification assertion failed!");
        }

        // 4. BENCHMARK SWEEP
        std::cout << "\n>>> [3] EMPIRICAL LATENCY & THROUGHPUT SWEEP (50 Iterations Warm Average)" << std::endl;

        for (uint32_t M : prompt_lengths) {
            std::cout << "\n" << std::string(112, '=') << std::endl;
            std::cout << " PROMPT LENGTH: M = " << M << " tokens | Dimensions: [M=" << M << ", K=" << K << ", N=" << N << "]" << std::endl;
            std::cout << std::string(112, '=') << std::endl;
            std::cout << std::left << std::setw(26) << "Kernel Configuration"
                      << std::setw(14) << "Latency (ms)"
                      << std::setw(16) << "Throughput"
                      << std::setw(16) << "Compute (TFLOPS)"
                      << std::setw(16) << "% FP16 Roofline"
                      << std::setw(14) << "vs llama.cpp"
                      << std::endl;
            std::cout << std::string(112, '-') << std::endl;

            size_t act_bytes = (size_t)M * K * sizeof(__fp16);
            size_t out_bytes = (size_t)M * N * sizeof(__fp16);
            id<MTLBuffer> bufA = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufC = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            generate_activations((__fp16*)[bufA contents], (size_t)M * K);

            double llama_ms = 0.0;

            for (const auto& kc : kernels) {
                auto run_kernel = [&]() -> double {
                    id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                    [encoder setComputePipelineState:pipelines[kc.name]];
                    [encoder setBuffer:bufA offset:0 atIndex:0];
                    [encoder setBuffer:bufB offset:0 atIndex:1];
                    [encoder setBuffer:bufC offset:0 atIndex:2];
                    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
                    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

                    if (kc.shmem_a_bytes > 0) [encoder setThreadgroupMemoryLength:kc.shmem_a_bytes atIndex:0];
                    if (kc.shmem_b_bytes > 0) [encoder setThreadgroupMemoryLength:kc.shmem_b_bytes atIndex:1];

                    NSUInteger tg_x = (N + kc.tg_n - 1) / kc.tg_n;
                    NSUInteger tg_y = (M + kc.tg_m - 1) / kc.tg_m;
                    [encoder dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(kc.tg_threads, 1, 1)];
                    [encoder endEncoding];

                    __block CFTimeInterval gpuStart = 0;
                    __block CFTimeInterval gpuEnd = 0;
                    [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        gpuStart = buffer.GPUStartTime;
                        gpuEnd = buffer.GPUEndTime;
                    }];

                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];
                    return (gpuEnd - gpuStart) * 1000.0;
                };

                // Warmup
                for (int w = 0; w < 10; w++) run_kernel();

                // Benchmarking
                const int iters = 50;
                double total_ms = 0.0;
                for (int i = 0; i < iters; i++) {
                    total_ms += run_kernel();
                }
                double avg_ms = total_ms / iters;

                if (kc.name.find("llama.cpp") != std::string::npos) {
                    llama_ms = avg_ms;
                }

                double total_flops = 2.0 * (double)M * (double)N * (double)K;
                double tflops = (total_flops / 1e12) / (avg_ms / 1000.0);
                double pct_roof = (tflops / peak_tflops) * 100.0;
                double tok_s = (double)M / (avg_ms / 1000.0);
                double speedup_vs_llama = (llama_ms > 0.0) ? (llama_ms / avg_ms) : 1.0;

                std::cout << std::left << std::setw(26) << kc.name
                          << std::setw(14) << std::fixed << std::setprecision(4) << avg_ms
                          << std::setw(16) << (std::to_string((int)tok_s) + " tok/s")
                          << std::setw(16) << std::fixed << std::setprecision(2) << tflops
                          << std::setw(16) << std::fixed << std::setprecision(2) << (std::to_string(pct_roof).substr(0, 5) + "%")
                          << std::setw(14) << std::fixed << std::setprecision(2) << (std::to_string(speedup_vs_llama).substr(0, 4) + "x")
                          << std::endl;
            }
        }
        std::cout << std::string(112, '=') << std::endl;
    }
    return 0;
}
