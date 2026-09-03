#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <string>

#import "core/metal/shader_loader.h"

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "No Metal device found." << std::endl;
            return 1;
        }

        id<MTLLibrary> lib = metal_llm::load_modular_metal_library(
            device,
            "tests/test_modular_kernels.metal",
            {"/Users/mohammedhossam/Documents/antigravity/wonderful-darwin", "include"}
        );
        if (!lib) {
            std::cerr << "load_modular_metal_library failed." << std::endl;
            return 1;
        }

        NSArray<NSString*>* names = @[
            @"test_modular_q4_0_gemm",
            @"test_modular_mlx_4bit_gemm",
            @"test_modular_q4_k_gemm",
            @"test_modular_ternary_1_58_gemm",
            @"test_modular_ternary_1_58_vec_gemm",
            @"test_modular_var_rate_affine_gemm",
            @"test_modular_exl3_gemm",
            @"test_modular_swiglu_q4_0",
            @"test_modular_flash_attn_d64",
            @"test_modular_flash_attn_d128"
        ];
        NSError* error = nil;
        for (NSString* name in names) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) {
                std::cerr << "Function not found: " << [name UTF8String] << std::endl;
                return 1;
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) {
                std::cerr << "Pipeline creation failed for: " << [name UTF8String] << std::endl;
                return 1;
            }
            std::cout << "Pipeline " << [name UTF8String] 
                      << " -> compiler maxTotalThreadsPerThreadgroup: " 
                      << [pso maxTotalThreadsPerThreadgroup] 
                      << " (static register pressure check)" << std::endl;
            if ([pso maxTotalThreadsPerThreadgroup] < 1024) {
                std::cerr << "Static register spill detected: maxTotalThreadsPerThreadgroup < 1024" << std::endl;
                return 1;
            }
        }

        std::cout << "ALL METAL HEADERS COMPILED WITH MAXIMUM STATIC THREADGROUP CAPACITY (1024 THREADS/THREADGROUP, <= 64 REGS/THREAD)!" << std::endl;
        std::cout << "Note: Numerical correctness & parity verified by test_kernel_parity & test_tier1_feature_coverage." << std::endl;
        return 0;
    }
}
