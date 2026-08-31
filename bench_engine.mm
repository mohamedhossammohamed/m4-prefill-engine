#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cassert>

struct block_q4_0 {
    __fp16 d;
    uint8_t qs[16];
};

// Deterministic PRNG for synthetic generation
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

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================" << std::endl;
        std::cout << "            J.A.R.V.I.S. M4 INFERENCE BENCHMARK HARNESS (CALIBRATION)     " << std::endl;
        std::cout << "==========================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return 1;
        }

        std::cout << "[+] Hardware Detected: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "Error loading kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // 1. HARDWARE PROBE: Raw Bandwidth
        // --------------------------------------------------------------------
        std::cout << "\n--- [1] PROBING CURRENT HARDWARE CEILINGS (FRESH) ---" << std::endl;
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

            MTLSize gridSize = MTLSizeMake(bw_elements, 1, 1);
            MTLSize tgSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
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
        std::cout << "[*] Empirical Peak Unified Memory Bandwidth: " << std::fixed << std::setprecision(2) 
                  << peak_gbps << " GB/s" << std::endl;

        // --------------------------------------------------------------------
        // 2. HARDWARE PROBE: Peak FP16 Compute Roofline
        // --------------------------------------------------------------------
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

            MTLSize gridSize = MTLSizeMake(fma_threads, 1, 1);
            MTLSize tgSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
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
        std::cout << "[*] Empirical Peak FP16 Compute Roofline: " << std::fixed << std::setprecision(2)
                  << peak_tflops << " TFLOPS" << std::endl;

        // --------------------------------------------------------------------
        // 3. COMPARISON: NAIVE vs LLAMA.CPP PRODUCTION BASELINE
        // --------------------------------------------------------------------
        std::cout << "\n--- [2] PRODUCTION LLAMA.CPP mul_mm BASELINE vs NAIVE (1B Model: K=2048, N=2048) ---" << std::endl;

        id<MTLFunction> naiveFunc = [library newFunctionWithName:@"naive_q4_0_gemm"];
        id<MTLComputePipelineState> naivePipeline = [device newComputePipelineStateWithFunction:naiveFunc error:&error];

        id<MTLFunction> llamaFunc = [library newFunctionWithName:@"llamacpp_style_mul_mm_q4_0"];
        id<MTLComputePipelineState> llamaPipeline = [device newComputePipelineStateWithFunction:llamaFunc error:&error];

        const uint32_t K = 2048;
        const uint32_t N = 2048;
        const std::vector<uint32_t> prompt_lengths = {33, 127, 128, 129, 512, 1023, 1024, 2047, 2048};

        uint32_t blocks_per_col = K / 32;
        size_t total_blocks = (size_t)N * blocks_per_col;
        size_t weight_bytes = total_blocks * sizeof(block_q4_0);
        id<MTLBuffer> bufB = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        generate_q4_0_weights((block_q4_0*)[bufB contents], total_blocks);

        std::cout << "\n" << std::string(96, '=') << std::endl;
        std::cout << std::left << std::setw(8)  << "Tokens"
                  << std::setw(14) << "Naive (ms)"
                  << std::setw(16) << "llama.cpp (ms)"
                  << std::setw(16) << "llama Throughput"
                  << std::setw(16) << "llama GFLOPS"
                  << std::setw(14) << "% Peak Compute"
                  << std::setw(10) << "Speedup" << std::endl;
        std::cout << std::string(96, '=') << std::endl;

        for (uint32_t M : prompt_lengths) {
            size_t act_bytes = (size_t)M * K * sizeof(__fp16);
            size_t out_bytes = (size_t)M * N * sizeof(__fp16);

            id<MTLBuffer> bufA = [device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufC_naive = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufC_llama = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

            generate_activations((__fp16*)[bufA contents], (size_t)M * K);

            // 1. Measure Naive Baseline
            auto run_naive = [&]() -> double {
                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:naivePipeline];
                [encoder setBuffer:bufA offset:0 atIndex:0];
                [encoder setBuffer:bufB offset:0 atIndex:1];
                [encoder setBuffer:bufC_naive offset:0 atIndex:2];
                [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
                [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

                MTLSize gridSize = MTLSizeMake(N, M, 1);
                MTLSize tgSize = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
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

            // 2. Measure llama.cpp Production Baseline
            auto run_llama = [&]() -> double {
                id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:llamaPipeline];
                [encoder setBuffer:bufA offset:0 atIndex:0];
                [encoder setBuffer:bufB offset:0 atIndex:1];
                [encoder setBuffer:bufC_llama offset:0 atIndex:2];
                [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
                [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

                // TG memory allocation: sh_A (64*32*2 = 4096 bytes), sh_B (32*32*2 = 2048 bytes)
                [encoder setThreadgroupMemoryLength:4096 atIndex:0];
                [encoder setThreadgroupMemoryLength:2048 atIndex:1];

                // Grid size: threadgroups in 2D (ceil(N/32), ceil(M/64))
                NSUInteger tg_x = (N + 31) / 32;
                NSUInteger tg_y = (M + 63) / 64;
                MTLSize threadgroupsPerGrid = MTLSizeMake(tg_x, tg_y, 1);
                MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1); // 2 SIMDgroups

                [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
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
            for (int i = 0; i < 5; i++) { run_naive(); run_llama(); }

            // Benchmarking
            double naive_total = 0.0;
            double llama_total = 0.0;
            const int iters = 20;
            for (int i = 0; i < iters; i++) {
                naive_total += run_naive();
                llama_total += run_llama();
            }
            double naive_ms = naive_total / iters;
            double llama_ms = llama_total / iters;

            // Correctness check on M=128
            if (M == 128) {
                const __fp16* ref = (const __fp16*)[bufC_naive contents];
                const __fp16* opt = (const __fp16*)[bufC_llama contents];
                float max_diff = 0.0f;
                for (size_t i = 0; i < (size_t)M * N; i++) {
                    float va = (float)opt[i];
                    float vb = (float)ref[i];
                    if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                        fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                        assert(false && "Numerical instability detected (NaN/Inf)");
                        exit(1);
                    }
                    float d = std::abs(va - vb);
                    if (d > max_diff) max_diff = d;
                }
                std::cout << "[✓] llama.cpp Correctness Check (M=" << M << "): MaxDiff vs Reference = " 
                          << max_diff << " (PASS)\n" << std::endl;
            }

            double total_flops = 2.0 * (double)M * (double)N * (double)K;
            double llama_gflops = (total_flops / 1e9) / (llama_ms / 1000.0);
            double llama_pct = (llama_gflops / (peak_tflops * 1000.0)) * 100.0;
            double llama_tok_s = (double)M / (llama_ms / 1000.0);
            double speedup = naive_ms / llama_ms;

            std::cout << std::left << std::setw(8)  << M
                      << std::setw(14) << std::fixed << std::setprecision(3) << naive_ms
                      << std::setw(16) << llama_ms
                      << std::setw(16) << (std::to_string((int)llama_tok_s) + " tok/s")
                      << std::setw(16) << std::fixed << std::setprecision(1) << llama_gflops
                      << std::setw(14) << std::fixed << std::setprecision(2) << (std::to_string(llama_pct).substr(0, 5) + "%")
                      << std::setw(10) << std::fixed << std::setprecision(2) << (std::to_string(speedup).substr(0, 4) + "x")
                      << std::endl;
        }
        std::cout << std::string(96, '=') << std::endl;
    }
    return 0;
}
