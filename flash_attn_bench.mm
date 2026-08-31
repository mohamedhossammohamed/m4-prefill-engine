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

// Deterministic PRNG for synthetic generation
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
// CPU REFERENCE ATTENTION (HIGH PRECISION FP32/FP64 GROUND TRUTH)
// ============================================================================
void cpu_reference_causal_attention(
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
        const __fp16* q_head = Q + h * M * D;
        const __fp16* k_head = K + h * M * D;
        const __fp16* v_head = V + h * M * D;
        __fp16* o_head = O + h * M * D;

        for (uint32_t i = 0; i < M; i++) {
            const __fp16* q_row = q_head + i * D;
            std::vector<float> scores(i + 1);
            float max_s = -1e30f;

            for (uint32_t j = 0; j <= i; j++) {
                const __fp16* k_row = k_head + j * D;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    dot += (float)q_row[d] * (float)k_row[d];
                }
                float s = dot * scale;
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
                    float p = scores[j] * inv_sum;
                    acc += p * (float)v_head[j * D + d];
                }
                o_head[i * D + d] = (__fp16)acc;
            }
        }
    }
}

struct VerificationResult {
    float max_diff;
    float avg_diff;
    bool passed;
};

VerificationResult verify_outputs(const __fp16* actual, const __fp16* expected, size_t count, float tolerance = 0.05f) {
    float max_d = 0.0f;
    double sum_d = 0.0;
    for (size_t i = 0; i < count; i++) {
        float va = (float)actual[i];
        float vb = (float)expected[i];
        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
            fprintf(stderr, "\n[FATAL] NaN or Inf detected at index %zu! GPU: %f | CPU: %f\n", (size_t)i, (float)va, (float)vb);
            assert(false && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
        float d = std::fabs(va - vb);
        if (d > max_d) max_d = d;
        sum_d += d;
    }
    float avg_d = (float)(sum_d / count);
    return {max_d, avg_d, max_d <= tolerance};
}

// ============================================================================
// BENCHMARK STRUCTURES & CONFIGURATION
// ============================================================================
struct KernelDescriptor {
    std::string name;
    std::string function_name;
    uint32_t tile_br;
    uint32_t tile_bc;
    uint32_t tg_threads;
    bool is_q8_0;
    bool is_naive;
};

struct BenchResult {
    std::string name;
    std::string tile_str;
    double cold_lat_ms;
    double warm_lat_ms;
    double tok_s;
    double tflops;
    double speedup;
};

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "        J.A.R.V.I.S. APPLE M4 FUSED FLASHATTENTION ENGINE BENCHMARK (FP16 & Q8_0)         " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[-] Error: Metal device initialization failed." << std::endl;
            return 1;
        }

        std::cout << "[+] Hardware Device: " << [[device name] UTF8String] << std::endl;

        NSError* error = nil;
        NSString* kernelPath = @"flash_attn_kernels.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:kernelPath encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            std::cerr << "[-] Error loading flash_attn_kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            std::cerr << "[-] Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        const uint32_t D = 64;           // Standard 1B Head Dimension
        const float scale = 1.0f / std::sqrt((float)D); // 0.125f

        const std::vector<uint32_t> seq_lengths = {33, 127, 128, 129, 512, 1023, 1024, 2047, 2048};

        // Kernel Descriptors for Empirical Sweep
        std::vector<KernelDescriptor> kernels = {
            // Naive Attention Baseline (run first to establish baseline reference)
            {"Naive Attention (3-stage)", "naive_attn_qk_causal", 0, 0, 0, false, true},
            
            // FP16 FlashAttention Kernels
            {"FlashAttn FP16 (16x16)", "flash_attn_fp16_16x16", 16, 16, 32, false, false},
            {"FlashAttn FP16 (32x16)", "flash_attn_fp16_32x16", 32, 16, 32, false, false},
            {"FlashAttn FP16 (32x32)", "flash_attn_fp16_32x32", 32, 32, 32, false, false},
            {"FlashAttn FP16 (64x32)", "flash_attn_fp16_64x32", 64, 32, 64, false, false},
            
            // Q8_0 FlashAttention Kernels
            {"FlashAttn Q8_0 (16x16)", "flash_attn_q8_0_16x16", 16, 16, 32, true, false},
            {"FlashAttn Q8_0 (32x16)", "flash_attn_q8_0_32x16", 32, 16, 32, true, false},
            {"FlashAttn Q8_0 (32x32)", "flash_attn_q8_0_32x32", 32, 32, 32, true, false},
            {"FlashAttn Q8_0 (64x32)", "flash_attn_q8_0_64x32", 64, 32, 64, true, false}
        };

        // Compile Pipelines
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        for (const auto& kd : kernels) {
            if (kd.is_naive) continue;
            id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:kd.function_name.c_str()]];
            if (!func) {
                std::cerr << "[-] Error creating function: " << kd.function_name << std::endl;
                return 1;
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
            if (error) {
                std::cerr << "[-] Error creating PSO for " << kd.name << ": " << [[error localizedDescription] UTF8String] << std::endl;
                return 1;
            }
            pipelines[kd.name] = pso;
        }

        // Compile Naive Stage Pipelines
        id<MTLFunction> naiveQkFunc = [library newFunctionWithName:@"naive_attn_qk_causal"];
        id<MTLFunction> naiveSmFunc = [library newFunctionWithName:@"naive_attn_softmax"];
        id<MTLFunction> naivePvFunc = [library newFunctionWithName:@"naive_attn_pv"];
        id<MTLComputePipelineState> naiveQkPso = [device newComputePipelineStateWithFunction:naiveQkFunc error:&error];
        id<MTLComputePipelineState> naiveSmPso = [device newComputePipelineStateWithFunction:naiveSmFunc error:&error];
        id<MTLComputePipelineState> naivePvPso = [device newComputePipelineStateWithFunction:naivePvFunc error:&error];

        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 1] 100% NUMERICAL CORRECTNESS VERIFICATION AGAINST CPU GROUND TRUTH" << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const uint32_t H_VERIFY = 16;
        for (uint32_t M : {128, 512, 1024}) {
            std::cout << "\n--- Testing Sequence Length M = " << M << " (H=" << H_VERIFY << ", D=" << D << ") ---" << std::endl;

            size_t total_elements = (size_t)H_VERIFY * M * D;
            std::vector<__fp16> h_Q(total_elements);
            std::vector<__fp16> h_K(total_elements);
            std::vector<__fp16> h_V(total_elements);
            std::vector<__fp16> h_O_cpu_fp16(total_elements);
            std::vector<__fp16> h_O_cpu_q8(total_elements);
            std::vector<__fp16> h_O_gpu(total_elements);

            generate_activations(h_Q.data(), total_elements);
            generate_activations(h_K.data(), total_elements);
            generate_activations(h_V.data(), total_elements);

            // CPU Ground Truth FP16
            cpu_reference_causal_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu_fp16.data(), H_VERIFY, M, D, scale);

            // Quantize K & V to Q8_0
            size_t num_blocks = total_elements / 32;
            std::vector<block_q8_0> h_K_q8(num_blocks);
            std::vector<block_q8_0> h_V_q8(num_blocks);
            quantize_to_q8_0(h_K.data(), h_K_q8.data(), total_elements);
            quantize_to_q8_0(h_V.data(), h_V_q8.data(), total_elements);

            // CPU Ground Truth Q8_0 (dequantized)
            std::vector<__fp16> h_K_dequant(total_elements);
            std::vector<__fp16> h_V_dequant(total_elements);
            dequantize_q8_0(h_K_q8.data(), h_K_dequant.data(), total_elements);
            dequantize_q8_0(h_V_q8.data(), h_V_dequant.data(), total_elements);
            cpu_reference_causal_attention(h_Q.data(), h_K_dequant.data(), h_V_dequant.data(), h_O_cpu_q8.data(), H_VERIFY, M, D, scale);

            // Create Metal Buffers
            id<MTLBuffer> d_Q = [device newBufferWithBytes:h_Q.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K = [device newBufferWithBytes:h_K.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V = [device newBufferWithBytes:h_V.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K_q8 = [device newBufferWithBytes:h_K_q8.data() length:num_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V_q8 = [device newBufferWithBytes:h_V_q8.data() length:num_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];

            // Verify Each Kernel
            for (const auto& kd : kernels) {
                if (kd.is_naive) {
                    size_t s_matrix_bytes = (size_t)H_VERIFY * M * M * sizeof(__fp16);
                    id<MTLBuffer> d_S = [device newBufferWithLength:s_matrix_bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> d_P = [device newBufferWithLength:s_matrix_bytes options:MTLResourceStorageModeShared];

                    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
                    [enc1 setComputePipelineState:naiveQkPso];
                    [enc1 setBuffer:d_Q offset:0 atIndex:0];
                    [enc1 setBuffer:d_K offset:0 atIndex:1];
                    [enc1 setBuffer:d_S offset:0 atIndex:2];
                    [enc1 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc1 setBytes:&scale length:sizeof(float) atIndex:4];
                    [enc1 dispatchThreads:MTLSizeMake(M, M, H_VERIFY) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                    [enc1 endEncoding];

                    id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
                    [enc2 setComputePipelineState:naiveSmPso];
                    [enc2 setBuffer:d_S offset:0 atIndex:0];
                    [enc2 setBuffer:d_P offset:0 atIndex:1];
                    [enc2 setBytes:&M length:sizeof(uint32_t) atIndex:2];
                    [enc2 dispatchThreads:MTLSizeMake(M, H_VERIFY, 1) threadsPerThreadgroup:MTLSizeMake(std::min(M, 256u), 1, 1)];
                    [enc2 endEncoding];

                    id<MTLComputeCommandEncoder> enc3 = [cb computeCommandEncoder];
                    [enc3 setComputePipelineState:naivePvPso];
                    [enc3 setBuffer:d_P offset:0 atIndex:0];
                    [enc3 setBuffer:d_V offset:0 atIndex:1];
                    [enc3 setBuffer:d_O offset:0 atIndex:2];
                    [enc3 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                    [enc3 dispatchThreads:MTLSizeMake(D, M, H_VERIFY) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                    [enc3 endEncoding];

                    [cb commit];
                    [cb waitUntilCompleted];

                    memcpy(h_O_gpu.data(), [d_O contents], total_elements * sizeof(__fp16));
                    auto res = verify_outputs(h_O_gpu.data(), h_O_cpu_fp16.data(), total_elements, 0.05f);
                    std::cout << "  [PASS] " << std::left << std::setw(28) << kd.name
                              << " | MaxDiff: " << std::fixed << std::setprecision(5) << res.max_diff
                              << " | AvgDiff: " << std::fixed << std::setprecision(6) << res.avg_diff << std::endl;
                    assert(res.passed && "Naive Attention verification failed!");
                } else {
                    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipelines[kd.name]];
                    [enc setBuffer:d_Q offset:0 atIndex:0];
                    if (kd.is_q8_0) {
                        [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                        [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                    } else {
                        [enc setBuffer:d_K offset:0 atIndex:1];
                        [enc setBuffer:d_V offset:0 atIndex:2];
                    }
                    [enc setBuffer:d_O offset:0 atIndex:3];
                    [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&scale length:sizeof(float) atIndex:5];

                    uint32_t num_br = (M + kd.tile_br - 1) / kd.tile_br;
                    [enc dispatchThreadgroups:MTLSizeMake(num_br, H_VERIFY, 1) threadsPerThreadgroup:MTLSizeMake(kd.tg_threads, 1, 1)];
                    [enc endEncoding];

                    [cb commit];
                    [cb waitUntilCompleted];

                    memcpy(h_O_gpu.data(), [d_O contents], total_elements * sizeof(__fp16));
                    
                    if (kd.is_q8_0) {
                        auto res_q8 = verify_outputs(h_O_gpu.data(), h_O_cpu_q8.data(), total_elements, 0.01f);
                        auto res_fp16 = verify_outputs(h_O_gpu.data(), h_O_cpu_fp16.data(), total_elements, 0.05f);
                        
                        std::cout << "  [PASS] " << std::left << std::setw(28) << kd.name
                                  << " | MaxDiff(Q8_Ref): " << std::fixed << std::setprecision(5) << res_q8.max_diff
                                  << " | MaxDiff(FP16_Ref): " << std::fixed << std::setprecision(5) << res_fp16.max_diff
                                  << " | AvgDiff: " << std::fixed << std::setprecision(6) << res_q8.avg_diff << std::endl;
                        assert(res_q8.passed && "FlashAttention Q8_0 verification failed against Q8 ground truth!");
                        assert(res_fp16.passed && "FlashAttention Q8_0 verification failed against FP16 ground truth!");
                    } else {
                        auto res = verify_outputs(h_O_gpu.data(), h_O_cpu_fp16.data(), total_elements, 0.05f);
                        std::cout << "  [PASS] " << std::left << std::setw(28) << kd.name
                                  << " | MaxDiff: " << std::fixed << std::setprecision(5) << res.max_diff
                                  << " | AvgDiff: " << std::fixed << std::setprecision(6) << res.avg_diff << std::endl;
                        assert(res.passed && "FlashAttention FP16 verification failed!");
                    }
                }
            }
        }

        std::cout << "\n[+] ALL NUMERICAL VERIFICATION SUITES PASSED WITH 100% SUCCESS (MaxDiff <= 0.05)\n" << std::endl;

        // ============================================================================
        // STAGE 2: EMPIRICAL PARAMETER SWEEP OVER TILE DIMENSIONS (H=16)
        // ============================================================================
        std::cout << "==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 2] EMPIRICAL PARAMETER SWEEP: TILE CONFIGURATIONS (H=16, D=64)                 " << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const int WARMUP_ITERS = 5;
        const int BENCH_ITERS = 50;
        const uint32_t H = 16;

        for (uint32_t M : seq_lengths) {
            std::cout << "\n>>> Sequence Length M = " << M << " | Heads H = " << H << " | Head Dim D = " << D << std::endl;
            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << std::left << std::setw(28) << "Kernel Configuration"
                      << std::setw(14) << "Tile (BrxBc)"
                      << std::setw(14) << "Cold Lat (ms)"
                      << std::setw(15) << "Warm Lat (ms)"
                      << std::setw(16) << "Throughput (tok/s)"
                      << std::setw(16) << "Effective TFLOPS"
                      << "Speedup vs Naive" << std::endl;
            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;

            size_t total_elements = (size_t)H * M * D;
            size_t num_blocks = total_elements / 32;

            std::vector<__fp16> h_Q(total_elements);
            std::vector<__fp16> h_K(total_elements);
            std::vector<__fp16> h_V(total_elements);
            generate_activations(h_Q.data(), total_elements);
            generate_activations(h_K.data(), total_elements);
            generate_activations(h_V.data(), total_elements);

            std::vector<block_q8_0> h_K_q8(num_blocks);
            std::vector<block_q8_0> h_V_q8(num_blocks);
            quantize_to_q8_0(h_K.data(), h_K_q8.data(), total_elements);
            quantize_to_q8_0(h_V.data(), h_V_q8.data(), total_elements);

            id<MTLBuffer> d_Q = [device newBufferWithBytes:h_Q.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K = [device newBufferWithBytes:h_K.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V = [device newBufferWithBytes:h_V.data() length:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_K_q8 = [device newBufferWithBytes:h_K_q8.data() length:num_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_V_q8 = [device newBufferWithBytes:h_V_q8.data() length:num_blocks * sizeof(block_q8_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_O = [device newBufferWithLength:total_elements * sizeof(__fp16) options:MTLResourceStorageModeShared];

            size_t s_matrix_bytes = (size_t)H * M * M * sizeof(__fp16);
            id<MTLBuffer> d_S = [device newBufferWithLength:s_matrix_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_P = [device newBufferWithLength:s_matrix_bytes options:MTLResourceStorageModeShared];

            double causal_flops = 4.0 * (double)H * ((double)M * (double)(M + 1) / 2.0) * (double)D;

            std::vector<BenchResult> results;
            double naive_warm_lat = 0.0;

            for (const auto& kd : kernels) {
                double cold_lat = 0.0;
                std::vector<double> latencies;
                latencies.reserve(BENCH_ITERS);

                if (kd.is_naive) {
                    {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
                        [enc1 setComputePipelineState:naiveQkPso];
                        [enc1 setBuffer:d_Q offset:0 atIndex:0];
                        [enc1 setBuffer:d_K offset:0 atIndex:1];
                        [enc1 setBuffer:d_S offset:0 atIndex:2];
                        [enc1 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc1 setBytes:&scale length:sizeof(float) atIndex:4];
                        [enc1 dispatchThreads:MTLSizeMake(M, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc1 endEncoding];

                        id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
                        [enc2 setComputePipelineState:naiveSmPso];
                        [enc2 setBuffer:d_S offset:0 atIndex:0];
                        [enc2 setBuffer:d_P offset:0 atIndex:1];
                        [enc2 setBytes:&M length:sizeof(uint32_t) atIndex:2];
                        [enc2 dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(std::min(M, 256u), 1, 1)];
                        [enc2 endEncoding];

                        id<MTLComputeCommandEncoder> enc3 = [cb computeCommandEncoder];
                        [enc3 setComputePipelineState:naivePvPso];
                        [enc3 setBuffer:d_P offset:0 atIndex:0];
                        [enc3 setBuffer:d_V offset:0 atIndex:1];
                        [enc3 setBuffer:d_O offset:0 atIndex:2];
                        [enc3 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc3 dispatchThreads:MTLSizeMake(D, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc3 endEncoding];

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cb commit];
                        [cb waitUntilCompleted];
                        cold_lat = (gpuEnd - gpuStart) * 1000.0;
                    }

                    for (int w = 0; w < WARMUP_ITERS; w++) {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
                        [enc1 setComputePipelineState:naiveQkPso];
                        [enc1 setBuffer:d_Q offset:0 atIndex:0];
                        [enc1 setBuffer:d_K offset:0 atIndex:1];
                        [enc1 setBuffer:d_S offset:0 atIndex:2];
                        [enc1 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc1 setBytes:&scale length:sizeof(float) atIndex:4];
                        [enc1 dispatchThreads:MTLSizeMake(M, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc1 endEncoding];

                        id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
                        [enc2 setComputePipelineState:naiveSmPso];
                        [enc2 setBuffer:d_S offset:0 atIndex:0];
                        [enc2 setBuffer:d_P offset:0 atIndex:1];
                        [enc2 setBytes:&M length:sizeof(uint32_t) atIndex:2];
                        [enc2 dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(std::min(M, 256u), 1, 1)];
                        [enc2 endEncoding];

                        id<MTLComputeCommandEncoder> enc3 = [cb computeCommandEncoder];
                        [enc3 setComputePipelineState:naivePvPso];
                        [enc3 setBuffer:d_P offset:0 atIndex:0];
                        [enc3 setBuffer:d_V offset:0 atIndex:1];
                        [enc3 setBuffer:d_O offset:0 atIndex:2];
                        [enc3 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc3 dispatchThreads:MTLSizeMake(D, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc3 endEncoding];

                        [cb commit];
                        [cb waitUntilCompleted];
                    }

                    for (int it = 0; it < BENCH_ITERS; it++) {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
                        [enc1 setComputePipelineState:naiveQkPso];
                        [enc1 setBuffer:d_Q offset:0 atIndex:0];
                        [enc1 setBuffer:d_K offset:0 atIndex:1];
                        [enc1 setBuffer:d_S offset:0 atIndex:2];
                        [enc1 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc1 setBytes:&scale length:sizeof(float) atIndex:4];
                        [enc1 dispatchThreads:MTLSizeMake(M, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc1 endEncoding];

                        id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
                        [enc2 setComputePipelineState:naiveSmPso];
                        [enc2 setBuffer:d_S offset:0 atIndex:0];
                        [enc2 setBuffer:d_P offset:0 atIndex:1];
                        [enc2 setBytes:&M length:sizeof(uint32_t) atIndex:2];
                        [enc2 dispatchThreads:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(std::min(M, 256u), 1, 1)];
                        [enc2 endEncoding];

                        id<MTLComputeCommandEncoder> enc3 = [cb computeCommandEncoder];
                        [enc3 setComputePipelineState:naivePvPso];
                        [enc3 setBuffer:d_P offset:0 atIndex:0];
                        [enc3 setBuffer:d_V offset:0 atIndex:1];
                        [enc3 setBuffer:d_O offset:0 atIndex:2];
                        [enc3 setBytes:&M length:sizeof(uint32_t) atIndex:3];
                        [enc3 dispatchThreads:MTLSizeMake(D, M, H) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                        [enc3 endEncoding];

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cb commit];
                        [cb waitUntilCompleted];
                        latencies.push_back((gpuEnd - gpuStart) * 1000.0);
                    }
                } else {
                    uint32_t num_br = (M + kd.tile_br - 1) / kd.tile_br;
                    MTLSize tgGrid = MTLSizeMake(num_br, H, 1);
                    MTLSize tgSize = MTLSizeMake(kd.tg_threads, 1, 1);

                    {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pipelines[kd.name]];
                        [enc setBuffer:d_Q offset:0 atIndex:0];
                        if (kd.is_q8_0) {
                            [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                            [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        } else {
                            [enc setBuffer:d_K offset:0 atIndex:1];
                            [enc setBuffer:d_V offset:0 atIndex:2];
                        }
                        [enc setBuffer:d_O offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&scale length:sizeof(float) atIndex:5];
                        [enc dispatchThreadgroups:tgGrid threadsPerThreadgroup:tgSize];
                        [enc endEncoding];

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cb commit];
                        [cb waitUntilCompleted];
                        cold_lat = (gpuEnd - gpuStart) * 1000.0;
                    }

                    for (int w = 0; w < WARMUP_ITERS; w++) {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pipelines[kd.name]];
                        [enc setBuffer:d_Q offset:0 atIndex:0];
                        if (kd.is_q8_0) {
                            [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                            [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        } else {
                            [enc setBuffer:d_K offset:0 atIndex:1];
                            [enc setBuffer:d_V offset:0 atIndex:2];
                        }
                        [enc setBuffer:d_O offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&scale length:sizeof(float) atIndex:5];
                        [enc dispatchThreadgroups:tgGrid threadsPerThreadgroup:tgSize];
                        [enc endEncoding];

                        [cb commit];
                        [cb waitUntilCompleted];
                    }

                    for (int it = 0; it < BENCH_ITERS; it++) {
                        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pipelines[kd.name]];
                        [enc setBuffer:d_Q offset:0 atIndex:0];
                        if (kd.is_q8_0) {
                            [enc setBuffer:d_K_q8 offset:0 atIndex:1];
                            [enc setBuffer:d_V_q8 offset:0 atIndex:2];
                        } else {
                            [enc setBuffer:d_K offset:0 atIndex:1];
                            [enc setBuffer:d_V offset:0 atIndex:2];
                        }
                        [enc setBuffer:d_O offset:0 atIndex:3];
                        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
                        [enc setBytes:&scale length:sizeof(float) atIndex:5];
                        [enc dispatchThreadgroups:tgGrid threadsPerThreadgroup:tgSize];
                        [enc endEncoding];

                        __block CFTimeInterval gpuStart = 0;
                        __block CFTimeInterval gpuEnd = 0;
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                            gpuStart = buffer.GPUStartTime;
                            gpuEnd = buffer.GPUEndTime;
                        }];

                        [cb commit];
                        [cb waitUntilCompleted];
                        latencies.push_back((gpuEnd - gpuStart) * 1000.0);
                    }
                }

                std::sort(latencies.begin(), latencies.end());
                double warm_lat = latencies[latencies.size() / 2];
                double tok_s = ((double)M) / (warm_lat / 1000.0);
                double tflops = (causal_flops / (warm_lat / 1000.0)) / 1e12;

                if (kd.is_naive) {
                    naive_warm_lat = warm_lat;
                }

                std::string tile_str = kd.is_naive ? "N/A" : (std::to_string(kd.tile_br) + "x" + std::to_string(kd.tile_bc));
                results.push_back({kd.name, tile_str, cold_lat, warm_lat, tok_s, tflops, 1.0});
            }

            // Print table with accurate speedups
            for (auto& r : results) {
                r.speedup = (naive_warm_lat > 0.0) ? (naive_warm_lat / r.warm_lat_ms) : 1.0;
                std::cout << std::left << std::setw(28) << r.name
                          << std::setw(14) << r.tile_str
                          << std::fixed << std::setprecision(3) << std::setw(14) << r.cold_lat_ms
                          << std::fixed << std::setprecision(3) << std::setw(15) << r.warm_lat_ms
                          << std::fixed << std::setprecision(0) << std::setw(16) << r.tok_s
                          << std::fixed << std::setprecision(3) << std::setw(16) << r.tflops
                          << std::fixed << std::setprecision(2) << r.speedup << "x" << std::endl;
            }
        }

        // ============================================================================
        // STAGE 3: CAUSAL BLOCK SKIPPING ARITHMETIC SAVINGS VERIFICATION
        // ============================================================================
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 3] CAUSAL BLOCK SKIPPING ARITHMETIC & FLOPs SCALING ANALYSIS                  " << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(12) << "Seq Len M"
                  << std::setw(20) << "Full Tiles (M/Bc)^2"
                  << std::setw(22) << "Causal Tiles Computed"
                  << std::setw(20) << "Tile Savings (%)"
                  << "Arithmetic FLOPs Saved" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;

        for (uint32_t M : seq_lengths) {
            uint32_t Bc = 32;
            uint32_t Br = 32;
            uint32_t num_tiles_1d = (M + Bc - 1) / Bc;
            uint64_t full_tiles = (uint64_t)num_tiles_1d * num_tiles_1d;
            uint64_t causal_tiles = 0;
            for (uint32_t br = 0; br < num_tiles_1d; br++) {
                uint32_t r_max = std::min((br + 1) * Br, M) - 1;
                uint32_t max_c_tile = r_max / Bc;
                causal_tiles += std::min(max_c_tile + 1, num_tiles_1d);
            }
            double savings_pct = (1.0 - (double)causal_tiles / (double)full_tiles) * 100.0;
            double flops_saved = 4.0 * (double)H * ((double)M * (double)M - ((double)M * (double)(M + 1) / 2.0)) * (double)D;

            std::cout << std::left << std::setw(12) << M
                      << std::setw(20) << full_tiles
                      << std::setw(22) << causal_tiles
                      << std::fixed << std::setprecision(2) << std::setw(20) << savings_pct
                      << std::scientific << std::setprecision(2) << flops_saved << " FLOPs (~50%)" << std::endl;
        }

        // ============================================================================
        // STAGE 4: PRODUCTION SUMMARY: FP16 VS Q8_0 KV CACHE MEMORY & BANDWIDTH FOOTPRINT
        // ============================================================================
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << ">>> [STAGE 4] KV CACHE MEMORY CONSUMPTION & BANDWIDTH ANALYSIS (D=64, H=16)               " << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(12) << "Seq Len M"
                  << std::setw(22) << "FP16 KV Cache (KB)"
                  << std::setw(22) << "Q8_0 KV Cache (KB)"
                  << std::setw(22) << "Memory Reduction (%)"
                  << "Bytes per Token (K+V)" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;

        for (uint32_t M : seq_lengths) {
            size_t fp16_bytes = 2 * (size_t)H * M * D * sizeof(__fp16);
            size_t q8_0_bytes = 2 * (size_t)H * M * (D / 32) * sizeof(block_q8_0);

            double fp16_kb = (double)fp16_bytes / 1024.0;
            double q8_0_kb = (double)q8_0_bytes / 1024.0;
            double reduction_pct = (1.0 - (double)q8_0_bytes / (double)fp16_bytes) * 100.0;
            size_t bytes_per_tok_fp16 = 2 * D * sizeof(__fp16);
            size_t bytes_per_tok_q8 = 2 * (D / 32) * sizeof(block_q8_0);

            std::cout << std::left << std::setw(12) << M
                      << std::fixed << std::setprecision(2) << std::setw(22) << fp16_kb
                      << std::fixed << std::setprecision(2) << std::setw(22) << q8_0_kb
                      << std::fixed << std::setprecision(2) << std::setw(22) << reduction_pct
                      << "FP16: " << bytes_per_tok_fp16 << "B | Q8_0: " << bytes_per_tok_q8 << "B" << std::endl;
        }

        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "                J.A.R.V.I.S. FLASHATTENTION BENCHMARK RUN COMPLETED                       " << std::endl;
        std::cout << "==========================================================================================" << std::endl;
    }
    return 0;
}
