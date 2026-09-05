#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "core/metal/shader_loader.h"
#include "models/bonsai-27b-ternary/runtime/bonsai_layer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

using namespace metal_llm;
using namespace metal_llm::bonsai;

// CPU Gold Reference for Gated-Delta-Net Recurrent Core
void cpu_gold_gated_delta_net(
    const __fp16* Q,
    const __fp16* K,
    const __fp16* V,
    const __fp16* g_in,
    const __fp16* b_in,
    float* state, // [H, D, D]
    __fp16* O,
    uint32_t M,
    uint32_t H,
    uint32_t D)
{
    float scale = 1.0f / std::sqrt((float)D);

    for (uint32_t h = 0; h < H; h++) {
        float* S = state + h * D * D;

        for (uint32_t t = 0; t < M; t++) {
            uint32_t tok_head_idx = (t * H + h) * D;
            const __fp16* q_t = Q + tok_head_idx;
            const __fp16* k_t = K + tok_head_idx;
            const __fp16* v_t = V + tok_head_idx;
            __fp16* o_t = O + tok_head_idx;

            float g_val = (float)g_in[t * H + h];
            float decay = std::exp(g_val);

            float b_val = (float)b_in[t * H + h];
            float beta_val = 1.0f / (1.0f + std::exp(-b_val));

            // S = S * decay
            for (uint32_t i = 0; i < D; i++) {
                for (uint32_t j = 0; j < D; j++) {
                    S[i * D + j] *= decay;
                }
            }

            // kv_mem = S * k_t
            std::vector<float> delta(D);
            for (uint32_t j = 0; j < D; j++) {
                float kv_mem_j = 0.0f;
                for (uint32_t i = 0; i < D; i++) {
                    kv_mem_j += S[i * D + j] * (float)k_t[i];
                }
                delta[j] = ((float)v_t[j] - kv_mem_j) * beta_val;
            }

            // S += k_t * delta^T
            for (uint32_t i = 0; i < D; i++) {
                float ki = (float)k_t[i];
                for (uint32_t j = 0; j < D; j++) {
                    S[i * D + j] += ki * delta[j];
                }
            }

            // o_t = S^T * (q_t * scale)
            for (uint32_t j = 0; j < D; j++) {
                float acc_o = 0.0f;
                for (uint32_t i = 0; i < D; i++) {
                    acc_o += S[i * D + j] * ((float)q_t[i] * scale);
                }
                o_t[j] = (__fp16)acc_o;
            }
        }
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " TESTING GATED-DELTA-NET LINEAR ATTENTION KERNEL & PARITY" << std::endl;
    std::cout << "=================================================================" << std::endl;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device != nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    assert(queue != nil);

    // Expand shader source including gated_delta_net.metal
    std::string shader_src = metal_llm::expand_shader_source(
        "include/metal/ops/gated_delta_net.metal",
        {".", "core", "include"}
    );

    NSError* error = nil;
    NSString* src_ns = [NSString stringWithUTF8String:shader_src.c_str()];
    id<MTLLibrary> lib = [device newLibraryWithSource:src_ns options:nil error:&error];
    if (!lib || error) {
        std::cerr << "Shader compilation failed: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }

    uint32_t M = 16;
    uint32_t H = 4;
    uint32_t D = 64;

    size_t qkv_bytes = M * H * D * sizeof(__fp16);
    size_t gh_bytes = M * H * sizeof(__fp16);
    size_t state_bytes = H * D * D * sizeof(float);

    id<MTLBuffer> buf_q = [device newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_k = [device newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_v = [device newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_g = [device newBufferWithLength:gh_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:gh_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_state_gpu = [device newBufferWithLength:state_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_o_gpu = [device newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];

    std::vector<float> state_cpu(H * D * D, 0.0f);
    std::vector<__fp16> o_cpu(M * H * D, 0.0f);

    __fp16* q_ptr = (__fp16*)[buf_q contents];
    __fp16* k_ptr = (__fp16*)[buf_k contents];
    __fp16* v_ptr = (__fp16*)[buf_v contents];
    __fp16* g_ptr = (__fp16*)[buf_g contents];
    __fp16* b_ptr = (__fp16*)[buf_b contents];

    for (size_t i = 0; i < M * H * D; i++) {
        q_ptr[i] = (__fp16)((float)((i % 17) - 8) * 0.05f);
        k_ptr[i] = (__fp16)((float)((i % 13) - 6) * 0.05f);
        v_ptr[i] = (__fp16)((float)((i % 19) - 9) * 0.05f);
    }
    for (size_t i = 0; i < M * H; i++) {
        g_ptr[i] = (__fp16)(-0.1f * ((i % 5) + 1));
        b_ptr[i] = (__fp16)(0.5f * ((i % 7) - 3));
    }
    memset([buf_state_gpu contents], 0, state_bytes);

    // Run CPU Gold Reference
    std::cout << "[*] Computing CPU Gold Reference..." << std::endl;
    cpu_gold_gated_delta_net(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_cpu.data(), o_cpu.data(), M, H, D);

    // Run GPU Kernel
    std::cout << "[*] Executing Metal GatedDeltaNet Recurrent Kernel..." << std::endl;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gated_delta_net_recurrent_core"];
    assert(fn != nil);
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    assert(pso != nil && error == nil);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_q offset:0 atIndex:0];
    [enc setBuffer:buf_k offset:0 atIndex:1];
    [enc setBuffer:buf_v offset:0 atIndex:2];
    [enc setBuffer:buf_g offset:0 atIndex:3];
    [enc setBuffer:buf_b offset:0 atIndex:4];
    [enc setBuffer:buf_state_gpu offset:0 atIndex:5];
    [enc setBuffer:buf_o_gpu offset:0 atIndex:6];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Verify Parity
    std::cout << "[*] Comparing GPU vs CPU output parity..." << std::endl;
    const __fp16* o_gpu = (const __fp16*)[buf_o_gpu contents];
    double max_err = 0.0;
    for (size_t i = 0; i < M * H * D; i++) {
        double diff = std::fabs((double)o_gpu[i] - (double)o_cpu[i]);
        if (diff > max_err) max_err = diff;
    }
    std::cout << "    -> Max Output Error: " << max_err << std::endl;
    assert(max_err < 1e-2); // FP16 tolerance

    // Check State Matrix Parity
    const float* s_gpu = (const float*)[buf_state_gpu contents];
    double max_state_err = 0.0;
    for (size_t i = 0; i < H * D * D; i++) {
        double diff = std::fabs((double)s_gpu[i] - (double)state_cpu[i]);
        if (diff > max_state_err) max_state_err = diff;
    }
    std::cout << "    -> Max State Error: " << max_state_err << std::endl;
    assert(max_state_err < 1e-3);

    std::cout << "GATED_DELTA_NET KERNEL PARITY TEST PASSED!" << std::endl;
    return 0;
}
