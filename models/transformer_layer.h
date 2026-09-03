#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <memory>
#include "src/router/quant_registry.h"

namespace metal_llm {

struct TransformerLayerConfig {
    uint32_t M = 128;          // Sequence length / prompt tokens
    uint32_t K = 2048;         // Hidden dimension
    uint32_t H = 32;           // Number of attention heads
    uint32_t D = 64;           // Head dimension (K = H * D)
    uint32_t N_mlp = 8192;     // Intermediate MLP dimension
    float attn_scale = 0.125f; // 1.0f / sqrt(D)
    QuantFormat weight_format = QUANT_Q4_0;
};

class TransformerLayerCoordinator {
public:
    TransformerLayerCoordinator(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const TransformerLayerConfig& config);

    ~TransformerLayerCoordinator() = default;

    // Load pipeline states from shader library
    bool initialize_pipelines(id<MTLLibrary> library);

    // Forward pass: executes a full transformer prefill layer on GPU
    void forward(
        id<MTLCommandBuffer> cmd_buf,
        id<MTLBuffer> X_in,
        id<MTLBuffer> W_q,
        id<MTLBuffer> W_k,
        id<MTLBuffer> W_v,
        id<MTLBuffer> W_o,
        id<MTLBuffer> W_gate,
        id<MTLBuffer> W_up,
        id<MTLBuffer> W_down,
        id<MTLBuffer> X_out);

    const TransformerLayerConfig& config() const { return config_; }

private:
    id<MTLDevice> device_;
    [[maybe_unused]] id<MTLCommandQueue> queue_;
    TransformerLayerConfig config_;

    // Pipeline states
    id<MTLComputePipelineState> pso_qkv_gemm_;
    id<MTLComputePipelineState> pso_flash_attn_;
    id<MTLComputePipelineState> pso_out_proj_;
    id<MTLComputePipelineState> pso_swiglu_;
    id<MTLComputePipelineState> pso_down_proj_;
    id<MTLComputePipelineState> pso_residual_add_;

    // Intermediates
    id<MTLBuffer> buf_q_;
    id<MTLBuffer> buf_k_;
    id<MTLBuffer> buf_v_;
    id<MTLBuffer> buf_attn_out_;
    id<MTLBuffer> buf_proj_out_;
    id<MTLBuffer> buf_swiglu_out_;
};

} // namespace metal_llm
