#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <memory>
#include "src/router/quant_registry.h"

namespace metal_llm {
namespace bonsai {

enum class LayerTopology {
    GATED_DELTA_NET, // 48 layers: linear attention (ssm/recurrent gated delta rule + causal conv1d + gated rmsnorm)
    FULL_GQA         // 16 layers: full grouped-query attention (QK-norm, Gated-Q, RoPE, D=256, H_q=24, H_kv=4)
};

// Returns whether layer index (0..63) is FULL_GQA or GATED_DELTA_NET
// GQA layers: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63] (every 4th layer)
inline LayerTopology get_layer_topology(int layer_idx) {
    if ((layer_idx + 1) % 4 == 0) {
        return LayerTopology::FULL_GQA;
    }
    return LayerTopology::GATED_DELTA_NET;
}

struct BonsaiLayerConfig {
    uint32_t layer_idx = 0;
    LayerTopology topology = LayerTopology::GATED_DELTA_NET;

    uint32_t M = 1;                     // Sequence length
    uint32_t hidden_dim = 5120;         // K = 5120
    uint32_t ffn_dim = 17408;           // N_mlp = 17408

    // GQA parameters (for 16 full-attention layers)
    uint32_t gqa_head_count = 24;       // H_q = 24
    uint32_t gqa_head_count_kv = 4;     // H_kv = 4
    uint32_t gqa_head_dim = 256;        // D = 256
    float gqa_attn_scale = 1.0f / 16.0f;// 1 / sqrt(256)

    // Gated-Delta-Net parameters (for 48 linear-attention layers)
    uint32_t ssm_inner_size = 6144;     // Inner value dim = 6144 (48 heads * 128)
    uint32_t ssm_head_dim = 128;        // D = 128
    uint32_t ssm_head_count = 48;       // H = 48
    uint32_t ssm_state_size = 128;      // State matrix: [48, 128, 128]
    uint32_t ssm_conv_kernel = 4;       // Conv1d kernel size = 4
    uint32_t ssm_conv_channels = 10240; // 2 * key_dim + value_dim = 2*2048 + 6144 = 10240

    QuantFormat weight_format = QUANT_PRISM_Q2_0;
};

// Coordinator for a single layer (routes dynamically according to config.topology)
class BonsaiLayerCoordinator {
public:
    BonsaiLayerCoordinator(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const BonsaiLayerConfig& config);

    ~BonsaiLayerCoordinator() = default;

    bool initialize_pipelines(id<MTLLibrary> library);

    // Forward pass for Gated-Delta-Net linear attention layer
    void forward_linear_attn(
        id<MTLCommandBuffer> cmd_buf,
        id<MTLBuffer> X_in,
        id<MTLBuffer> W_qkv,
        id<MTLBuffer> W_z,
        id<MTLBuffer> W_b,
        id<MTLBuffer> W_a,
        id<MTLBuffer> W_ssm_out,
        id<MTLBuffer> ssm_conv1d_weight,
        id<MTLBuffer> ssm_a,
        id<MTLBuffer> ssm_dt_bias,
        id<MTLBuffer> ssm_norm_weight,
        id<MTLBuffer> recurrent_state, // [48, 128, 128] FP32
        id<MTLBuffer> conv_state,      // [3, 10240] FP16
        id<MTLBuffer> W_gate,
        id<MTLBuffer> W_up,
        id<MTLBuffer> W_down,
        id<MTLBuffer> X_out);

    // Forward pass for Full GQA layer
    void forward_gqa(
        id<MTLCommandBuffer> cmd_buf,
        id<MTLBuffer> X_in,
        id<MTLBuffer> W_q,
        id<MTLBuffer> W_k,
        id<MTLBuffer> W_v,
        id<MTLBuffer> W_o,
        id<MTLBuffer> q_norm_weight,
        id<MTLBuffer> k_norm_weight,
        id<MTLBuffer> kv_cache_k,
        id<MTLBuffer> kv_cache_v,
        uint32_t kv_offset,
        id<MTLBuffer> W_gate,
        id<MTLBuffer> W_up,
        id<MTLBuffer> W_down,
        id<MTLBuffer> X_out);

    const BonsaiLayerConfig& config() const { return config_; }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    BonsaiLayerConfig config_;

    // Pipelines
    id<MTLComputePipelineState> pso_gemm_;
    id<MTLComputePipelineState> pso_swiglu_;
    id<MTLComputePipelineState> pso_residual_add_;

    // GDN-specific pipelines
    id<MTLComputePipelineState> pso_conv1d_;
    id<MTLComputePipelineState> pso_head_l2norm_;
    id<MTLComputePipelineState> pso_compute_gate_;
    id<MTLComputePipelineState> pso_recurrent_core_;
    id<MTLComputePipelineState> pso_gated_rmsnorm_;

    // GQA-specific pipelines
    id<MTLComputePipelineState> pso_flash_attn_;
    id<MTLComputePipelineState> pso_q_gate_mul_;

    // Scratch intermediate buffers
    id<MTLBuffer> buf_qkv_;
    id<MTLBuffer> buf_conv_out_;
    id<MTLBuffer> buf_z_;
    id<MTLBuffer> buf_b_;
    id<MTLBuffer> buf_a_;
    id<MTLBuffer> buf_g_;
    id<MTLBuffer> buf_attn_out_;
    id<MTLBuffer> buf_proj_out_;
    id<MTLBuffer> buf_swiglu_out_;
};

} // namespace bonsai
} // namespace metal_llm
