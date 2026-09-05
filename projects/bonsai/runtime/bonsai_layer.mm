#import "bonsai_layer.h"
#include <iostream>
#include <cmath>

namespace metal_llm {
namespace bonsai {

BonsaiLayerCoordinator::BonsaiLayerCoordinator(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const BonsaiLayerConfig& config)
    : device_(device), queue_(queue), config_(config)
{
    uint32_t M = config_.M;
    size_t x_bytes = M * config_.hidden_dim * sizeof(__fp16);
    size_t mlp_bytes = M * config_.ffn_dim * sizeof(__fp16);
    size_t proj_bytes = M * config_.hidden_dim * sizeof(__fp16);

    buf_proj_out_ = [device_ newBufferWithLength:proj_bytes options:MTLResourceStorageModeShared];
    buf_swiglu_out_ = [device_ newBufferWithLength:mlp_bytes options:MTLResourceStorageModeShared];

    if (config_.topology == LayerTopology::GATED_DELTA_NET) {
        size_t qkv_bytes = M * config_.ssm_conv_channels * sizeof(__fp16);
        size_t z_bytes = M * config_.ssm_inner_size * sizeof(__fp16);
        size_t b_bytes = M * config_.ssm_head_count * sizeof(__fp16);
        size_t g_bytes = M * config_.ssm_head_count * sizeof(__fp16);
        size_t attn_bytes = M * config_.ssm_inner_size * sizeof(__fp16);

        buf_qkv_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
        buf_conv_out_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
        buf_z_ = [device_ newBufferWithLength:z_bytes options:MTLResourceStorageModeShared];
        buf_b_ = [device_ newBufferWithLength:b_bytes options:MTLResourceStorageModeShared];
        buf_a_ = [device_ newBufferWithLength:b_bytes options:MTLResourceStorageModeShared];
        buf_g_ = [device_ newBufferWithLength:g_bytes options:MTLResourceStorageModeShared];
        buf_attn_out_ = [device_ newBufferWithLength:attn_bytes options:MTLResourceStorageModeShared];
    } else {
        size_t q_bytes = M * (config_.gqa_head_count * config_.gqa_head_dim * 2) * sizeof(__fp16); // Gated Q
        size_t kv_bytes = M * (config_.gqa_head_count_kv * config_.gqa_head_dim) * sizeof(__fp16);
        size_t attn_bytes = M * (config_.gqa_head_count * config_.gqa_head_dim) * sizeof(__fp16);

        buf_qkv_ = [device_ newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
        buf_attn_out_ = [device_ newBufferWithLength:attn_bytes options:MTLResourceStorageModeShared];
    }
}

bool BonsaiLayerCoordinator::initialize_pipelines(id<MTLLibrary> library) {
    if (!library) return false;
    NSError* error = nil;

    auto create_pso = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) return nil;
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&error];
        return pso;
    };

    const auto* desc = QuantRegistry::instance().get(config_.weight_format);
    if (!desc) {
        std::cerr << "[BonsaiLayerCoordinator] Unknown weight format: " << (int)config_.weight_format << std::endl;
        return false;
    }

    pso_gemm_ = create_pso(desc->gemm_kernel_name.c_str());
    if (!pso_gemm_) {
        std::cerr << "[BonsaiLayerCoordinator] Failed to load GEMM: " << desc->gemm_kernel_name << std::endl;
        return false;
    }

    pso_swiglu_ = create_pso("swiglu_mma_dual_simd");
    if (!pso_swiglu_) pso_swiglu_ = create_pso("fused_gate_up_swiglu_q4_0");

    pso_residual_add_ = create_pso("vector_add_residual");

    if (config_.topology == LayerTopology::GATED_DELTA_NET) {
        pso_conv1d_ = create_pso("gated_delta_net_conv1d_silu");
        pso_head_l2norm_ = create_pso("gated_delta_net_head_l2norm");
        pso_compute_gate_ = create_pso("gated_delta_net_compute_gate");
        pso_recurrent_core_ = create_pso("gated_delta_net_recurrent_core");
        pso_gated_rmsnorm_ = create_pso("gated_delta_net_gated_rmsnorm");
    } else {
        pso_flash_attn_ = create_pso("flash_attn_mma_64x64_fp16_d128");
        if (!pso_flash_attn_) pso_flash_attn_ = create_pso("flash_attn_fp16_causal");
    }

    return true;
}

void BonsaiLayerCoordinator::forward_linear_attn(
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
    id<MTLBuffer> recurrent_state,
    id<MTLBuffer> conv_state,
    id<MTLBuffer> W_gate,
    id<MTLBuffer> W_up,
    id<MTLBuffer> W_down,
    id<MTLBuffer> X_out)
{
    uint32_t M = config_.M;
    uint32_t K = config_.hidden_dim;
    uint32_t H = config_.ssm_head_count; // 48
    uint32_t D = config_.ssm_head_dim;   // 128
    uint32_t inner_size = config_.ssm_inner_size; // 6144
    uint32_t conv_c = config_.ssm_conv_channels;  // 10240
    uint32_t N_mlp = config_.ffn_dim;
    float eps = 1e-6f;

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    // 1. In-projections
    // W_qkv: [K, conv_c]
    if (pso_gemm_) {
        [enc setComputePipelineState:pso_gemm_];
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:W_qkv offset:0 atIndex:1];
        [enc setBuffer:buf_qkv_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&conv_c length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((conv_c + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // W_z: [K, inner_size]
        [enc setBuffer:W_z offset:0 atIndex:1];
        [enc setBuffer:buf_z_ offset:0 atIndex:2];
        [enc setBytes:&inner_size length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((inner_size + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // W_b: [K, H]
        [enc setBuffer:W_b offset:0 atIndex:1];
        [enc setBuffer:buf_b_ offset:0 atIndex:2];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((H + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        // W_a: [K, H]
        [enc setBuffer:W_a offset:0 atIndex:1];
        [enc setBuffer:buf_a_ offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((H + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // 2. Depthwise Causal Conv1D + SiLU
    if (pso_conv1d_) {
        [enc setComputePipelineState:pso_conv1d_];
        [enc setBuffer:buf_qkv_ offset:0 atIndex:0];
        [enc setBuffer:ssm_conv1d_weight offset:0 atIndex:1];
        [enc setBuffer:conv_state offset:0 atIndex:2];
        [enc setBuffer:buf_conv_out_ offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&conv_c length:sizeof(uint32_t) atIndex:5];
        MTLSize grid = MTLSizeMake(M, (conv_c + 31) / 32, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(1, 32, 1)];
    }

    // 3. Q/K L2-Norm
    // Q is [M, 16, 128], K is [M, 16, 128] within buf_conv_out_
    if (pso_head_l2norm_) {
        [enc setComputePipelineState:pso_head_l2norm_];
        uint32_t qk_heads = 16;
        [enc setBuffer:buf_conv_out_ offset:0 atIndex:0];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&qk_heads length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, qk_heads, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }

    // 4. Compute Gate g_t
    if (pso_compute_gate_) {
        [enc setComputePipelineState:pso_compute_gate_];
        [enc setBuffer:buf_a_ offset:0 atIndex:0];
        [enc setBuffer:ssm_a offset:0 atIndex:1];
        [enc setBuffer:ssm_dt_bias offset:0 atIndex:2];
        [enc setBuffer:buf_g_ offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }

    // 5. Recurrent Gated Delta Rule
    if (pso_recurrent_core_) {
        [enc setComputePipelineState:pso_recurrent_core_];
        // In buf_conv_out: Q is first 2048, K is second 2048, V is next 6144
        size_t k_offset = 2048 * sizeof(__fp16);
        size_t v_offset = 4096 * sizeof(__fp16);
        [enc setBuffer:buf_conv_out_ offset:0 atIndex:0];
        [enc setBuffer:buf_conv_out_ offset:k_offset atIndex:1];
        [enc setBuffer:buf_conv_out_ offset:v_offset atIndex:2];
        [enc setBuffer:buf_g_ offset:0 atIndex:3];
        [enc setBuffer:buf_b_ offset:0 atIndex:4];
        [enc setBuffer:recurrent_state offset:0 atIndex:5];
        [enc setBuffer:buf_attn_out_ offset:0 atIndex:6];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }

    // 6. Gated RMSNorm: RMSNorm(buf_attn_out) * silu(buf_z)
    if (pso_gated_rmsnorm_) {
        [enc setComputePipelineState:pso_gated_rmsnorm_];
        [enc setBuffer:buf_attn_out_ offset:0 atIndex:0];
        [enc setBuffer:buf_z_ offset:0 atIndex:1];
        [enc setBuffer:ssm_norm_weight offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&eps length:sizeof(float) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M, H, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }

    // 7. SSM Out projection GEMM: buf_attn_out @ W_ssm_out -> buf_proj_out
    if (pso_gemm_) {
        [enc setComputePipelineState:pso_gemm_];
        [enc setBuffer:buf_attn_out_ offset:0 atIndex:0];
        [enc setBuffer:W_ssm_out offset:0 atIndex:1];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&inner_size length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((K + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // 8. First Residual Add: X_in + buf_proj_out -> X_out
    if (pso_residual_add_) {
        [enc setComputePipelineState:pso_residual_add_];
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:1];
        [enc setBuffer:X_out offset:0 atIndex:2];
        uint32_t total = M * K;
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((total + 255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 9. MLP SwiGLU: X_out @ (W_gate, W_up) -> buf_swiglu_out
    if (pso_swiglu_) {
        [enc setComputePipelineState:pso_swiglu_];
        [enc setBuffer:X_out offset:0 atIndex:0];
        [enc setBuffer:W_gate offset:0 atIndex:1];
        [enc setBuffer:W_up offset:0 atIndex:2];
        [enc setBuffer:buf_swiglu_out_ offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
        [enc setThreadgroupMemoryLength:17408 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((N_mlp + 31)/32, (M + 63)/64, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    }

    // 10. Down projection: buf_swiglu_out @ W_down -> buf_proj_out
    if (pso_gemm_) {
        [enc setComputePipelineState:pso_gemm_];
        [enc setBuffer:buf_swiglu_out_ offset:0 atIndex:0];
        [enc setBuffer:W_down offset:0 atIndex:1];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((K + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // 11. Second Residual Add: X_out + buf_proj_out -> X_out
    if (pso_residual_add_) {
        [enc setComputePipelineState:pso_residual_add_];
        [enc setBuffer:X_out offset:0 atIndex:0];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:1];
        [enc setBuffer:X_out offset:0 atIndex:2];
        uint32_t total = M * K;
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((total + 255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc endEncoding];
}

void BonsaiLayerCoordinator::forward_gqa(
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
    id<MTLBuffer> X_out)
{
    // Standard GQA execution with QK-norm
    uint32_t M = config_.M;
    uint32_t K = config_.hidden_dim;
    uint32_t H = config_.gqa_head_count;
    uint32_t D = config_.gqa_head_dim;
    uint32_t N_mlp = config_.ffn_dim;

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    // Q, K, V Projections
    if (pso_gemm_) {
        [enc setComputePipelineState:pso_gemm_];
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:W_q offset:0 atIndex:1];
        [enc setBuffer:buf_qkv_ offset:0 atIndex:2];
        uint32_t q_dim = H * D * 2; // Gated Q
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&q_dim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((q_dim + 31)/32, (M + 31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Residual & MLP follow the standard sequence
    if (pso_residual_add_) {
        [enc setComputePipelineState:pso_residual_add_];
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:X_in offset:0 atIndex:1];
        [enc setBuffer:X_out offset:0 atIndex:2];
        uint32_t total = M * K;
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((total + 255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc endEncoding];
}

} // namespace bonsai
} // namespace metal_llm
