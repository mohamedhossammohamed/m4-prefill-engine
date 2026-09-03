#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "core/metal/shader_loader.h"
#include "core/memory/page_allocator.h"
#include "tests/e2e/test_common.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace metal_llm;

struct KernelParityHarness {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> monolith_lib;
    id<MTLLibrary> modular_lib;

    bool init() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return false;
        queue = [device newCommandQueue];

        NSError* error = nil;
        NSString* mono_src = [NSString stringWithContentsOfFile:@"quant_router_kernels.metal"
                                                       encoding:NSUTF8StringEncoding error:&error];
        if (error || !mono_src) {
            std::cerr << "Failed to read quant_router_kernels.metal" << std::endl;
            return false;
        }

        monolith_lib = [device newLibraryWithSource:mono_src options:nil error:&error];
        if (error || !monolith_lib) {
            std::cerr << "Failed to compile monolithic library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        modular_lib = metal_llm::load_modular_metal_library(device, "tests/test_modular_kernels.metal", {".", "core", "include"});
        if (!modular_lib) {
            std::cerr << "Failed to compile modular library via load_modular_metal_library" << std::endl;
            return false;
        }

        return true;
    }

    void run_kernel(
        id<MTLLibrary> lib,
        const std::string& name,
        id<MTLBuffer> bufA,
        id<MTLBuffer> bufB,
        id<MTLBuffer> bufC,
        uint32_t M,
        uint32_t N,
        uint32_t K)
    {
        NSError* error = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        assert(fn != nil);
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        assert(pso != nil && error == nil);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];

        // 16KB threadgroup memory
        [enc setThreadgroupMemoryLength:16384 atIndex:0];

        MTLSize grid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
        MTLSize tgroup = MTLSizeMake(128, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }
};

template <typename TBlock, typename TGen>
bool verify_kernel_parity(
    KernelParityHarness& h,
    QuantFormat fmt,
    const std::string& mono_name,
    const std::string& mod_name,
    TGen generator,
    uint32_t block_step,
    uint32_t M,
    uint32_t N,
    uint32_t K)
{
    size_t act_bytes = (size_t)M * K * sizeof(__fp16);
    size_t out_bytes = (size_t)M * N * sizeof(__fp16);
    size_t weight_bytes = compute_quant_weight_bytes(fmt, (size_t)N * K);

    id<MTLBuffer> bufA = [h.device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [h.device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC_mono = [h.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC_mod  = [h.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

    generate_activations((__fp16*)[bufA contents], (size_t)M * K);
    generator((TBlock*)[bufB contents], (size_t)N * (K / block_step));

    // Clear outputs
    std::memset([bufC_mono contents], 0, out_bytes);
    std::memset([bufC_mod  contents], 0, out_bytes);

    // Run Monolith
    h.run_kernel(h.monolith_lib, mono_name, bufA, bufB, bufC_mono, M, N, K);

    // Run Modular
    h.run_kernel(h.modular_lib, mod_name, bufA, bufB, bufC_mod, M, N, K);

    // Compare Outputs
    const __fp16* p_mono = (const __fp16*)[bufC_mono contents];
    const __fp16* p_mod  = (const __fp16*)[bufC_mod  contents];

    size_t bad_idx = 0;
    if (!verify_finite(p_mono, M * N, &bad_idx) || !verify_finite(p_mod, M * N, &bad_idx)) {
        std::cerr << "    [-] Tripwire failed: non-finite output at index " << bad_idx << std::endl;
        return false;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        float diff = std::fabs((float)p_mono[i] - (float)p_mod[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Bit-exact identity check between Monolithic and Modular implementations
    if (max_diff > 0.0f) {
        std::cerr << "    [-] Parity failure: max_diff = " << max_diff << " (expected 0.0f)" << std::endl;
        return false;
    }

    return true;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " PRE/POST REFACTOR KERNEL PARITY & BIT-EXACTNESS VERIFICATION" << std::endl;
    std::cout << "=================================================================" << std::endl;

    KernelParityHarness h;
    if (!h.init()) {
        std::cerr << "Harness initialization failed!" << std::endl;
        return 1;
    }

    const uint32_t K = 512;
    const uint32_t N = 256;
    int passed = 0;
    int total = 0;

    // 1. Q4_0
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [Q4_0] M=" << M << "... ";
        if (verify_kernel_parity<block_q4_0>(h, QUANT_Q4_0, "quant_router_gemm_q4_0_64x64", "test_modular_q4_0_gemm", generate_q4_0_weights, 32, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    // 2. MLX_4BIT
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [MLX_4BIT] M=" << M << "... ";
        if (verify_kernel_parity<block_mlx_4bit>(h, QUANT_MLX_4BIT, "quant_router_gemm_mlx_4bit_64x64", "test_modular_mlx_4bit_gemm", generate_mlx_4bit_weights, 32, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    // 3. Q4_K
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [Q4_K] M=" << M << "... ";
        if (verify_kernel_parity<block_q4_K>(h, QUANT_Q4_K, "quant_router_gemm_q4_k_64x64", "test_modular_q4_k_gemm", generate_q4_k_weights, 256, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    // 4. TERNARY_1_58
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [TERNARY_1_58] M=" << M << "... ";
        if (verify_kernel_parity<block_ternary_1_58>(h, QUANT_TERNARY_1_58, "quant_router_gemm_ternary_1_58_64x64", "test_modular_ternary_1_58_gemm", generate_ternary_1_58_weights, 32, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    // 5. VAR_RATE_AFFINE
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [VAR_RATE_AFFINE] M=" << M << "... ";
        if (verify_kernel_parity<block_var_rate_affine>(h, QUANT_VAR_RATE_AFFINE, "quant_router_gemm_var_rate_affine_64x64", "test_modular_var_rate_affine_gemm", generate_var_rate_affine_weights, 256, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    // 6. EXL3
    for (uint32_t M : {33, 127, 128, 129, 512}) {
        total++;
        std::cout << "  RUN Parity [EXL3] M=" << M << "... ";
        if (verify_kernel_parity<block_exl3>(h, QUANT_EXL3, "quant_router_gemm_exl3_64x64", "test_modular_exl3_gemm", generate_exl3_weights, 256, M, N, K)) {
            std::cout << "BIT-EXACT (diff=0.000000)" << std::endl;
            passed++;
        }
    }

    std::cout << "=================================================================" << std::endl;
    std::cout << " SUMMARY: Pre/Post Refactor Parity" << std::endl;
    std::cout << "  Total Configurations: " << total << std::endl;
    std::cout << "  Passed (Bit-Exact):   " << passed << std::endl;
    std::cout << "  Failed:               " << (total - passed) << std::endl;
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
