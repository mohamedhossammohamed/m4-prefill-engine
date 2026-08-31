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
    uint32_t tg_threads;
    uint32_t shmem_bytes;
    bool is_llamacpp;
    bool is_naive;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "  J.A.R.V.I.S. M4 MEMORY QUEUE SATURATION & DOUBLE-BUFFERED PREFILL ENGINE BENCHMARK      " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[-] Error: Metal is not supported on this device." << std::endl;
            return 1;
        }

        std::cout << "[+] Hardware Platform: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"pipelined_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[-] Error loading pipelined_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[-] Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // --------------------------------------------------------------------
        // 1. HARDWARE PROBING: Peak Memory Bandwidth & FP16 Compute Roofline
        // --------------------------------------------------------------------
        std::cout << "\n>>> [1] PROBING HARDWARE LIMITS (M4 MEMORY CONTROLLERS & FP16 FMAs)" << std::endl;
        
        // Memory Bandwidth Probe
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
        std::cout << "    [+] Empirical Peak Unified Memory Bandwidth: " << std::fixed << std::setprecision(2)
                  << peak_gbps << " GB/s" << std::endl;

        // FP16 Roofline Probe
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
        std::cout << "    [+] Empirical Peak FP16 Compute Roofline:   " << std::fixed << std::setprecision(2)
                  << peak_tflops << " TFLOPS" << std::endl;

        // --------------------------------------------------------------------
        // KERNEL CONFIGURATIONS REGISTRY
        // --------------------------------------------------------------------
        std::vector<KernelConfig> all_kernels = {
            // Baselines
            {"Naive Reference",            "naive_q4_0_gemm",           1,   1,   256, 0,     false, true},
            {"llama.cpp (Drawer 8x4)",     "llamacpp_style_mul_mm_q4_0", 64,  32,  64,  4096,  true,  false},
            
            // Tile sweep (Double buffered)
            {"Pipe-Double (32x32)",        "pipe_gemm_32x32_double",    32,  32,  32,  4096,  false, false},
            {"Pipe-Double (64x32)",        "pipe_gemm_64x32_double",    64,  32,  64,  8192,  false, false},
            {"Pipe-Double (64x64)",        "pipe_gemm_64x64_double",    64,  64,  128, 8192,  false, false},
            {"Pipe-Double (128x32)",       "pipe_gemm_128x32_double",   128, 32,  128, 16384, false, false},

            // Single buffered counterparts (for delta comparison)
            {"Pipe-Single (32x32)",        "pipe_gemm_32x32_single",    32,  32,  32,  2048,  false, false},
            {"Pipe-Single (64x32)",        "pipe_gemm_64x32_single",    64,  32,  64,  4096,  false, false},
            {"Pipe-Single (64x64)",        "pipe_gemm_64x64_single",    64,  64,  128, 4096,  false, false},
            {"Pipe-Single (128x32)",       "pipe_gemm_128x32_single",   128, 32,  128, 8192,  false, false},

            // Vector load width variations on 64x32
            {"Pipe-Vec64  (64x32)",        "pipe_gemm_64x32_vec64",     64,  32,  64,  8192,  false, false},
            {"Pipe-Vec32  (64x32)",        "pipe_gemm_64x32_vec32",     64,  32,  64,  8192,  false, false},

            // 2D Register-tiled Double Buffered
            {"Pipe-Double-2D (64x32)",     "pipe_gemm_64x32_2d_double", 64,  32,  64,  8192,  false, false}
        };

        // Create Pipelines
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        for (const auto& kc : all_kernels) {
            id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:kc.function_name.c_str()]];
            if (!fn) {
                std::cerr << "[-] Error: Function not found: " << kc.function_name << std::endl;
                return 1;
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (error) {
                std::cerr << "[-] Error compiling pipeline for " << kc.function_name << ": "
                          << [[error localizedDescription] UTF8String] << std::endl;
                return 1;
            }
            pipelines[kc.name] = pso;
        }

        // Runner lambda
        auto dispatch_kernel = [&](const KernelConfig& kc, id<MTLBuffer> bufA, id<MTLBuffer> bufB, id<MTLBuffer> bufC, uint32_t M, uint32_t N, uint32_t K) -> double {
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipelines[kc.name]];
            [encoder setBuffer:bufA offset:0 atIndex:0];
            [encoder setBuffer:bufB offset:0 atIndex:1];
            [encoder setBuffer:bufC offset:0 atIndex:2];
            [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

            if (kc.is_naive) {
                MTLSize gridSize = MTLSizeMake(N, M, 1);
                MTLSize tgSize = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            } else if (kc.is_llamacpp) {
                [encoder setThreadgroupMemoryLength:4096 atIndex:0];
                [encoder setThreadgroupMemoryLength:2048 atIndex:1];
                NSUInteger tg_x = (N + 31) / 32;
                NSUInteger tg_y = (M + 63) / 64;
                MTLSize threadgroupsPerGrid = MTLSizeMake(tg_x, tg_y, 1);
                MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
                [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            } else {
                [encoder setThreadgroupMemoryLength:kc.shmem_bytes atIndex:0];
                NSUInteger tg_x = (N + kc.tile_n - 1) / kc.tile_n;
                NSUInteger tg_y = (M + kc.tile_m - 1) / kc.tile_m;
                MTLSize threadgroupsPerGrid = MTLSizeMake(tg_x, tg_y, 1);
                MTLSize threadsPerThreadgroup = MTLSizeMake(kc.tg_threads, 1, 1);
                [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            }

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

        // --------------------------------------------------------------------
        // 2. NUMERICAL CORRECTNESS VERIFICATION (M=128, K=2048, N=2048)
        // --------------------------------------------------------------------
        std::cout << "\n>>> [2] NUMERICAL CORRECTNESS VERIFICATION (CPU Gold Standard Reference, MaxDiff <= 0.05)" << std::endl;
        const uint32_t val_M = 128;
        const uint32_t val_K = 2048;
        const uint32_t val_N = 2048;

        size_t val_act_bytes = (size_t)val_M * val_K * sizeof(__fp16);
        size_t val_out_bytes = (size_t)val_M * val_N * sizeof(__fp16);
        size_t val_blocks = (size_t)val_N * (val_K / 32);
        size_t val_wt_bytes = val_blocks * sizeof(block_q4_0);

        id<MTLBuffer> val_bufA = [device newBufferWithLength:val_act_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> val_bufB = [device newBufferWithLength:val_wt_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> val_bufC_gpu = [device newBufferWithLength:val_out_bytes options:MTLResourceStorageModeShared];
        std::vector<__fp16> val_C_cpu(val_M * val_N);

        generate_activations((__fp16*)[val_bufA contents], val_M * val_K);
        generate_q4_0_weights((block_q4_0*)[val_bufB contents], val_blocks);

        std::cout << "    [*] Computing CPU Reference GEMM (M=128, K=2048, N=2048)..." << std::flush;
        cpu_reference_gemm(
            (const __fp16*)[val_bufA contents],
            (const block_q4_0*)[val_bufB contents],
            val_C_cpu.data(),
            val_M, val_N, val_K
        );
        std::cout << " Done." << std::endl;

        bool all_passed = true;
        for (const auto& kc : all_kernels) {
            memset([val_bufC_gpu contents], 0, val_out_bytes);
            dispatch_kernel(kc, val_bufA, val_bufB, val_bufC_gpu, val_M, val_N, val_K);

            const __fp16* gpu_out = (const __fp16*)[val_bufC_gpu contents];
            float max_diff = 0.0f;
            for (size_t i = 0; i < (size_t)val_M * val_N; i++) {
                float va = (float)gpu_out[i];
                float vb = (float)val_C_cpu[i];
                if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
                    fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
                    assert(false && "Numerical instability detected (NaN/Inf)");
                    exit(1);
                }
                float diff = std::abs(va - vb);
                if (diff > max_diff) max_diff = diff;
            }

            bool passed = (max_diff <= 0.05f);
            if (!passed) all_passed = false;

            std::cout << "    [" << (passed ? "PASS" : "FAIL") << "] "
                      << std::left << std::setw(28) << kc.name
                      << " MaxDiff vs CPU = " << std::fixed << std::setprecision(5) << max_diff
                      << (passed ? "  (<= 0.05)" : "  (VIOLATION)") << std::endl;
        }

        if (!all_passed) {
            std::cerr << "[-] Error: One or more kernels failed numerical correctness validation." << std::endl;
            return 1;
        }
        std::cout << "[✓] 100% Numerical Correctness Verified Across All Pipelined Variants." << std::endl;

        // --------------------------------------------------------------------
        // 3. PARAMETER SWEEP BENCHMARKS (M = 1024, K=2048, N=2048)
        // --------------------------------------------------------------------
        const uint32_t sweep_M = 1024;
        const uint32_t K = 2048;
        const uint32_t N = 2048;

        size_t sw_act_bytes = (size_t)sweep_M * K * sizeof(__fp16);
        size_t sw_out_bytes = (size_t)sweep_M * N * sizeof(__fp16);
        size_t sw_blocks = (size_t)N * (K / 32);
        size_t sw_wt_bytes = sw_blocks * sizeof(block_q4_0);

        id<MTLBuffer> sw_bufA = [device newBufferWithLength:sw_act_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sw_bufB = [device newBufferWithLength:sw_wt_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sw_bufC = [device newBufferWithLength:sw_out_bytes options:MTLResourceStorageModeShared];

        generate_activations((__fp16*)[sw_bufA contents], sweep_M * K);
        generate_q4_0_weights((block_q4_0*)[sw_bufB contents], sw_blocks);

        auto benchmark_kernel = [&](const KernelConfig& kc, uint32_t M_len, id<MTLBuffer> bA, id<MTLBuffer> bB, id<MTLBuffer> bC, double& out_cold_ms) -> double {
            // Cold latency
            out_cold_ms = dispatch_kernel(kc, bA, bB, bC, M_len, N, K);

            // Warmup
            for (int i = 0; i < 5; i++) {
                dispatch_kernel(kc, bA, bB, bC, M_len, N, K);
            }

            // 50-iteration Warm latency
            const int iters = 50;
            double total_ms = 0.0;
            for (int i = 0; i < iters; i++) {
                total_ms += dispatch_kernel(kc, bA, bB, bC, M_len, N, K);
            }
            return total_ms / iters;
        };

        // --- SWEEP A: Tile Configurations (32x32, 64x32, 64x64, 128x32) ---
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "  PARAMETER SWEEP 1: TILE CONFIGURATION (M=1024, K=2048, N=2048, Double-Buffered)" << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(24) << "Tile Config"
                  << std::setw(12) << "Threads"
                  << std::setw(14) << "Cold (ms)"
                  << std::setw(14) << "Warm (ms)"
                  << std::setw(18) << "Throughput"
                  << std::setw(16) << "Compute"
                  << std::setw(14) << "% Roofline" << std::endl;
        std::cout << std::string(98, '-') << std::endl;

        std::vector<std::string> tile_names = {
            "Pipe-Double (32x32)",
            "Pipe-Double (64x32)",
            "Pipe-Double (64x64)",
            "Pipe-Double (128x32)",
            "Pipe-Double-2D (64x32)"
        };

        for (const auto& name : tile_names) {
            const auto& kc = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == name; });
            double cold_ms = 0.0;
            double warm_ms = benchmark_kernel(kc, sweep_M, sw_bufA, sw_bufB, sw_bufC, cold_ms);
            double total_flops = 2.0 * (double)sweep_M * (double)N * (double)K;
            double tflops = (total_flops / 1e12) / (warm_ms / 1000.0);
            double pct_roof = (tflops / peak_tflops) * 100.0;
            double tok_s = (double)sweep_M / (warm_ms / 1000.0);

            std::cout << std::left << std::setw(24) << kc.name
                  << std::setw(12) << kc.tg_threads
                  << std::setw(14) << std::fixed << std::setprecision(4) << cold_ms
                  << std::setw(14) << warm_ms
                  << std::setw(18) << (std::to_string((int)tok_s) + " tok/s")
                  << std::setw(16) << (std::to_string(tflops).substr(0, 5) + " TFLOPS")
                  << std::setw(14) << (std::to_string(pct_roof).substr(0, 5) + " %")
                  << std::endl;
        }

        // --- SWEEP B: Vector Load Widths (32-bit vs 64-bit vs 128-bit) ---
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "  PARAMETER SWEEP 2: VECTOR LOAD WIDTHS (64x32 Tile, M=1024, K=2048, N=2048)" << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(24) << "Load Width"
                  << std::setw(14) << "Cold (ms)"
                  << std::setw(14) << "Warm (ms)"
                  << std::setw(18) << "Throughput"
                  << std::setw(16) << "Compute"
                  << std::setw(14) << "Speedup vs 32b" << std::endl;
        std::cout << std::string(98, '-') << std::endl;

        std::vector<std::string> vec_names = {
            "Pipe-Vec32  (64x32)",
            "Pipe-Vec64  (64x32)",
            "Pipe-Double (64x32)" // vec128
        };

        double base_vec32_ms = 0.0;
        for (const auto& name : vec_names) {
            const auto& kc = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == name; });
            double cold_ms = 0.0;
            double warm_ms = benchmark_kernel(kc, sweep_M, sw_bufA, sw_bufB, sw_bufC, cold_ms);
            if (name == "Pipe-Vec32  (64x32)") base_vec32_ms = warm_ms;
            double total_flops = 2.0 * (double)sweep_M * (double)N * (double)K;
            double tflops = (total_flops / 1e12) / (warm_ms / 1000.0);
            double tok_s = (double)sweep_M / (warm_ms / 1000.0);
            double spd = base_vec32_ms / warm_ms;

            std::string label = (name == "Pipe-Double (64x32)") ? "128-bit (half8 firehose)" : ((name == "Pipe-Vec64  (64x32)") ? "64-bit  (half4 vector)" : "32-bit  (half2 scalar)");

            std::cout << std::left << std::setw(24) << label
                      << std::setw(14) << std::fixed << std::setprecision(4) << cold_ms
                      << std::setw(14) << warm_ms
                      << std::setw(18) << (std::to_string((int)tok_s) + " tok/s")
                      << std::setw(16) << (std::to_string(tflops).substr(0, 5) + " TFLOPS")
                      << std::setw(14) << (std::to_string(spd).substr(0, 4) + "x")
                      << std::endl;
        }

        // --- SWEEP C: Single vs Double Buffering Ping-Pong Delta ---
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "  PARAMETER SWEEP 3: SINGLE vs DOUBLE BUFFERING DELTA (M=1024, K=2048, N=2048)" << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(18) << "Tile Shape"
                  << std::setw(16) << "Single-Buf (ms)"
                  << std::setw(16) << "Double-Buf (ms)"
                  << std::setw(18) << "Single TFLOPS"
                  << std::setw(18) << "Double TFLOPS"
                  << std::setw(14) << "Double Delta" << std::endl;
        std::cout << std::string(98, '-') << std::endl;

        std::vector<std::pair<std::string, std::string>> buf_pairs = {
            {"Pipe-Single (32x32)",  "Pipe-Double (32x32)"},
            {"Pipe-Single (64x32)",  "Pipe-Double (64x32)"},
            {"Pipe-Single (64x64)",  "Pipe-Double (64x64)"},
            {"Pipe-Single (128x32)", "Pipe-Double (128x32)"}
        };

        for (const auto& p : buf_pairs) {
            const auto& kc_s = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == p.first; });
            const auto& kc_d = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == p.second; });
            
            double c_s = 0.0, c_d = 0.0;
            double ms_s = benchmark_kernel(kc_s, sweep_M, sw_bufA, sw_bufB, sw_bufC, c_s);
            double ms_d = benchmark_kernel(kc_d, sweep_M, sw_bufA, sw_bufB, sw_bufC, c_d);

            double total_flops = 2.0 * (double)sweep_M * (double)N * (double)K;
            double tf_s = (total_flops / 1e12) / (ms_s / 1000.0);
            double tf_d = (total_flops / 1e12) / (ms_d / 1000.0);
            double delta_pct = ((ms_s - ms_d) / ms_s) * 100.0;
            double spd = ms_s / ms_d;

            std::string shape_name = p.first.substr(12);

            std::cout << std::left << std::setw(18) << shape_name
                      << std::setw(16) << std::fixed << std::setprecision(4) << ms_s
                      << std::setw(16) << ms_d
                      << std::setw(18) << (std::to_string(tf_s).substr(0, 5) + " TFLOPS")
                      << std::setw(18) << (std::to_string(tf_d).substr(0, 5) + " TFLOPS")
                      << std::setw(14) << ("+" + std::to_string(delta_pct).substr(0, 4) + "% (" + std::to_string(spd).substr(0, 4) + "x)")
                      << std::endl;
        }

        // --------------------------------------------------------------------
        // 4. PRODUCTION BENCHMARK SWEEP (M in [33, 127, 128, 129, 512, 1023, 1024, 2047, 2048])
        // --------------------------------------------------------------------
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "  PRODUCTION BENCHMARK SWEEP: OPTIMIZED PIPELINE vs LLAMA.CPP vs NAIVE (1B Model Shape)   " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const std::vector<uint32_t> prod_prompt_lengths = {33, 127, 128, 129, 512, 1023, 1024, 2047, 2048};

        const auto& naive_kc   = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == "Naive Reference"; });
        const auto& llama_kc   = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == "llama.cpp (Drawer 8x4)"; });
        const auto& opt32_kc   = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == "Pipe-Double (32x32)"; });
        const auto& opt64_kc   = *std::find_if(all_kernels.begin(), all_kernels.end(), [&](const KernelConfig& k){ return k.name == "Pipe-Double (64x32)"; });

        for (uint32_t M_len : prod_prompt_lengths) {
            size_t p_act_bytes = (size_t)M_len * K * sizeof(__fp16);
            size_t p_out_bytes = (size_t)M_len * N * sizeof(__fp16);
            size_t p_blocks = (size_t)N * (K / 32);
            size_t p_wt_bytes = p_blocks * sizeof(block_q4_0);

            id<MTLBuffer> p_bufA = [device newBufferWithLength:p_act_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> p_bufB = [device newBufferWithLength:p_wt_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> p_bufC = [device newBufferWithLength:p_out_bytes options:MTLResourceStorageModeShared];

            generate_activations((__fp16*)[p_bufA contents], M_len * K);
            generate_q4_0_weights((block_q4_0*)[p_bufB contents], p_blocks);

            double naive_cold = 0.0, llama_cold = 0.0, opt32_cold = 0.0, opt64_cold = 0.0;
            double naive_warm = benchmark_kernel(naive_kc, M_len, p_bufA, p_bufB, p_bufC, naive_cold);
            double llama_warm = benchmark_kernel(llama_kc, M_len, p_bufA, p_bufB, p_bufC, llama_cold);
            double opt32_warm = benchmark_kernel(opt32_kc, M_len, p_bufA, p_bufB, p_bufC, opt32_cold);
            double opt64_warm = benchmark_kernel(opt64_kc, M_len, p_bufA, p_bufB, p_bufC, opt64_cold);

            double total_flops = 2.0 * (double)M_len * (double)N * (double)K;
            
            double opt32_tflops = (total_flops / 1e12) / (opt32_warm / 1000.0);
            double opt32_pct_roof = (opt32_tflops / peak_tflops) * 100.0;
            double opt32_tok_s = (double)M_len / (opt32_warm / 1000.0);

            double opt64_tflops = (total_flops / 1e12) / (opt64_warm / 1000.0);
            double opt64_pct_roof = (opt64_tflops / peak_tflops) * 100.0;
            double opt64_tok_s = (double)M_len / (opt64_warm / 1000.0);

            double llama_tflops = (total_flops / 1e12) / (llama_warm / 1000.0);
            double llama_tok_s = (double)M_len / (llama_warm / 1000.0);

            std::cout << "\n--------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " [PROMPT M = " << std::setw(4) << M_len << " tokens] Dimensions: [M=" << M_len << ", K=" << K << ", N=" << N << "]" << std::endl;
            std::cout << "--------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << std::left << std::setw(26) << "Engine / Kernel"
                      << std::setw(12) << "Cold (ms)"
                      << std::setw(12) << "Warm (ms)"
                      << std::setw(18) << "Throughput"
                      << std::setw(14) << "TFLOPS"
                      << std::setw(14) << "% Roofline"
                      << std::setw(20) << "Speedup" << std::endl;
            std::cout << std::string(116, '-') << std::endl;

            std::cout << std::left << std::setw(26) << "Naive Baseline"
                      << std::setw(12) << std::fixed << std::setprecision(4) << naive_cold
                      << std::setw(12) << naive_warm
                      << std::setw(18) << (std::to_string((int)((double)M_len / (naive_warm / 1000.0))) + " tok/s")
                      << std::setw(14) << (std::to_string((total_flops / 1e12) / (naive_warm / 1000.0)).substr(0, 5))
                      << std::setw(14) << (std::to_string((((total_flops / 1e12) / (naive_warm / 1000.0)) / peak_tflops) * 100.0).substr(0, 5) + " %")
                      << std::setw(20) << "1.00x (Ref)" << std::endl;

            std::cout << std::left << std::setw(26) << "llama.cpp Production"
                      << std::setw(12) << std::fixed << std::setprecision(4) << llama_cold
                      << std::setw(12) << llama_warm
                      << std::setw(18) << (std::to_string((int)llama_tok_s) + " tok/s")
                      << std::setw(14) << (std::to_string(llama_tflops).substr(0, 5))
                      << std::setw(14) << (std::to_string((llama_tflops / peak_tflops) * 100.0).substr(0, 5) + " %")
                      << std::setw(20) << (std::to_string(naive_warm / llama_warm).substr(0, 4) + "x (vs Naive)") << std::endl;

            std::cout << std::left << std::setw(26) << "Pipelined-32x32 (1-SIMD)"
                      << std::setw(12) << std::fixed << std::setprecision(4) << opt32_cold
                      << std::setw(12) << opt32_warm
                      << std::setw(18) << (std::to_string((int)opt32_tok_s) + " tok/s")
                      << std::setw(14) << (std::to_string(opt32_tflops).substr(0, 5))
                      << std::setw(14) << (std::to_string(opt32_pct_roof).substr(0, 5) + " %")
                      << std::setw(20) << (std::to_string(llama_warm / opt32_warm).substr(0, 4) + "x (vs llama) / " + std::to_string(naive_warm / opt32_warm).substr(0, 4) + "x (vs Naive)") << std::endl;

            std::cout << std::left << std::setw(26) << "Pipelined-64x32 (2-SIMD)"
                      << std::setw(12) << std::fixed << std::setprecision(4) << opt64_cold
                      << std::setw(12) << opt64_warm
                      << std::setw(18) << (std::to_string((int)opt64_tok_s) + " tok/s")
                      << std::setw(14) << (std::to_string(opt64_tflops).substr(0, 5))
                      << std::setw(14) << (std::to_string(opt64_pct_roof).substr(0, 5) + " %")
                      << std::setw(20) << (std::to_string(llama_warm / opt64_warm).substr(0, 4) + "x (vs llama) / " + std::to_string(naive_warm / opt64_warm).substr(0, 4) + "x (vs Naive)") << std::endl;
        }

        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << " [✓] ALL BENCHMARK PHASES & PARAMETER SWEEPS COMPLETED SUCCESSFULLY." << std::endl;
        std::cout << "==========================================================================================" << std::endl;
    }
    return 0;
}
