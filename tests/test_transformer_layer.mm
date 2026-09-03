#import "models/transformer_layer.h"
#include "core/memory/page_allocator.h"
#include "core/metrology/tripwires.h"
#include "tests/e2e/test_common.h"
#include <iostream>
#include <cassert>

using namespace metal_llm;
using namespace core::metrology;

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " TESTING COMPOSABLE TRANSFORMER LAYER COORDINATOR" << std::endl;
    std::cout << "=================================================================" << std::endl;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device != nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    assert(queue != nil);

    // Load library
    NSError* error = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"quant_router_kernels.metal"
                                              encoding:NSUTF8StringEncoding error:&error];
    assert(error == nil && src != nil);
    id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
    assert(error == nil && library != nil);

    TransformerLayerConfig config;
    config.M = 64;
    config.K = 256;
    config.H = 4;
    config.D = 64;
    config.N_mlp = 512;
    config.attn_scale = 1.0f / std::sqrt(64.0f);
    config.weight_format = core::memory::QUANT_Q4_0;

    std::cout << "[*] Instantiating TransformerLayerCoordinator..." << std::endl;
    TransformerLayerCoordinator layer(device, queue, config);

    std::cout << "[*] Initializing pipelines from shader library..." << std::endl;
    bool init_ok = layer.initialize_pipelines(library);
    assert(init_ok);
    std::cout << "    -> Pipelines initialized successfully." << std::endl;

    // Allocate synthetic buffers
    size_t x_bytes = config.M * config.K * sizeof(__fp16);
    size_t w_attn_bytes = compute_quant_weight_bytes(::QUANT_Q4_0, config.K * config.K);
    size_t w_mlp_bytes = compute_quant_weight_bytes(::QUANT_Q4_0, config.K * config.N_mlp);

    id<MTLBuffer> X_in = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_q = [device newBufferWithLength:w_attn_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_k = [device newBufferWithLength:w_attn_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_v = [device newBufferWithLength:w_attn_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_o = [device newBufferWithLength:w_attn_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_gate = [device newBufferWithLength:w_mlp_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_up = [device newBufferWithLength:w_mlp_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_down = [device newBufferWithLength:w_mlp_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> X_out = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];

    generate_activations((__fp16*)[X_in contents], config.M * config.K);
    generate_q4_0_weights((::block_q4_0*)[W_q contents], (config.K * config.K) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_k contents], (config.K * config.K) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_v contents], (config.K * config.K) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_o contents], (config.K * config.K) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_gate contents], (config.K * config.N_mlp) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_up contents], (config.K * config.N_mlp) / 32);
    generate_q4_0_weights((::block_q4_0*)[W_down contents], (config.K * config.N_mlp) / 32);

    std::cout << "[*] Executing forward pass on GPU command buffer..." << std::endl;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    layer.forward(cmd, X_in, W_q, W_k, W_v, W_o, W_gate, W_up, W_down, X_out);
    [cmd commit];
    [cmd waitUntilCompleted];

    size_t bad_idx = 0;
    bool finite = verify_finite((const __fp16*)[X_out contents], config.M * config.K, &bad_idx);
    assert(finite);
    std::cout << "    -> Output validated finite, zero NaN/Inf detected." << std::endl;

    std::cout << "TRANSFORMER_LAYER_COORDINATOR TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
