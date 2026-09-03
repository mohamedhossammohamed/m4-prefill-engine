#import "transformer_layer.h"
#include <iostream>
#include <cmath>

namespace metal_llm {

TransformerLayerCoordinator::TransformerLayerCoordinator(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const TransformerLayerConfig& config)
    : device_(device), queue_(queue), config_(config)
{
    // Allocate intermediate buffers
    size_t attn_dim = config_.H * config_.D;
    size_t qkv_bytes = config_.M * attn_dim * sizeof(__fp16);
    size_t mlp_bytes = config_.M * config_.N_mlp * sizeof(__fp16);
    size_t out_bytes = config_.M * config_.K * sizeof(__fp16);

    buf_q_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    buf_k_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    buf_v_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    buf_attn_out_ = [device_ newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
    buf_proj_out_ = [device_ newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    buf_swiglu_out_ = [device_ newBufferWithLength:mlp_bytes options:MTLResourceStorageModeShared];
}

bool TransformerLayerCoordinator::initialize_pipelines(id<MTLLibrary> library) {
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
        std::cerr << "[TransformerLayerCoordinator] Unsupported format: " << (int)config_.weight_format << std::endl;
        return false;
    }

    // Direct head or standard GEMM
    pso_qkv_gemm_ = create_pso(desc->head_gemm_kernel_name.c_str());
    if (!pso_qkv_gemm_) {
        pso_qkv_gemm_ = create_pso(desc->gemm_kernel_name.c_str());
    }

    // FlashAttention
    if (config_.D == 64) {
        pso_flash_attn_ = create_pso("flash_attn_mma_64x64_fp16_d64");
        if (!pso_flash_attn_) pso_flash_attn_ = create_pso("flash_attn_fp16_causal");
    } else {
        pso_flash_attn_ = create_pso("flash_attn_mma_64x64_fp16_d128");
        if (!pso_flash_attn_) pso_flash_attn_ = create_pso("flash_attn_fp16_causal_d128");
    }

    // Out-projection GEMM
    pso_out_proj_ = create_pso(desc->gemm_kernel_name.c_str());

    // SwiGLU Dual-SIMD
    pso_swiglu_ = create_pso("swiglu_mma_dual_simd");
    if (!pso_swiglu_) pso_swiglu_ = create_pso("fused_gate_up_swiglu_q4_0");

    // Down projection GEMM
    pso_down_proj_ = create_pso(desc->gemm_kernel_name.c_str());

    // Residual add
    pso_residual_add_ = create_pso("vector_add_residual");

    return (pso_qkv_gemm_ != nil);
}

void TransformerLayerCoordinator::forward(
    id<MTLCommandBuffer> cmd_buf,
    id<MTLBuffer> X_in,
    id<MTLBuffer> W_q,
    id<MTLBuffer> W_k,
    id<MTLBuffer> W_v,
    id<MTLBuffer> W_o,
    id<MTLBuffer> W_gate,
    id<MTLBuffer> W_up,
    id<MTLBuffer> W_down,
    id<MTLBuffer> X_out)
{
    uint32_t M = config_.M;
    uint32_t K = config_.K;
    uint32_t H = config_.H;
    uint32_t D = config_.D;
    uint32_t attn_dim = H * D;
    uint32_t N_mlp = config_.N_mlp;
    float attn_scale = config_.attn_scale;

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    // 1. QKV Projections
    if (pso_qkv_gemm_) {
        [enc setComputePipelineState:pso_qkv_gemm_];
        
        // Q projection
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:W_q offset:0 atIndex:1];
        [enc setBuffer:buf_q_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];

        MTLSize tg_size = MTLSizeMake(32, 1, 1);
        MTLSize grid_size = MTLSizeMake((attn_dim + 31) / 32, (M + 31) / 32, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];

        // K projection
        [enc setBuffer:W_k offset:0 atIndex:1];
        [enc setBuffer:buf_k_ offset:0 atIndex:2];
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];

        // V projection
        [enc setBuffer:W_v offset:0 atIndex:1];
        [enc setBuffer:buf_v_ offset:0 atIndex:2];
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    }

    // 2. FlashAttention
    if (pso_flash_attn_) {
        [enc setComputePipelineState:pso_flash_attn_];
        [enc setBuffer:buf_q_ offset:0 atIndex:0];
        [enc setBuffer:buf_k_ offset:0 atIndex:1];
        [enc setBuffer:buf_v_ offset:0 atIndex:2];
        [enc setBuffer:buf_attn_out_ offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&attn_scale length:sizeof(float) atIndex:5];
        [enc setThreadgroupMemoryLength:16384 atIndex:0];

        MTLSize tg_size = MTLSizeMake(32, 4, 1);
        MTLSize grid_size = MTLSizeMake((M + 63) / 64, H, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    }

    // 3. Out-projection GEMM: buf_attn_out @ W_o -> buf_proj_out
    if (pso_out_proj_) {
        [enc setComputePipelineState:pso_out_proj_];
        [enc setBuffer:buf_attn_out_ offset:0 atIndex:0];
        [enc setBuffer:W_o offset:0 atIndex:1];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&attn_dim length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];

        MTLSize tg_size = MTLSizeMake(32, 1, 1);
        MTLSize grid_size = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    }

    // 4. Residual Add: X_in + buf_proj_out -> X_out
    if (pso_residual_add_) {
        [enc setComputePipelineState:pso_residual_add_];
        [enc setBuffer:X_in offset:0 atIndex:0];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:1];
        [enc setBuffer:X_out offset:0 atIndex:2];
        uint32_t total = M * K;
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:3];
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }

    // 5. SwiGLU MLP
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

        MTLSize tg_size = MTLSizeMake(32, 4, 1);
        MTLSize grid_size = MTLSizeMake((N_mlp + 31) / 32, (M + 63) / 64, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    }

    // 6. Down-projection GEMM: buf_swiglu_out @ W_down -> buf_proj_out
    if (pso_down_proj_) {
        [enc setComputePipelineState:pso_down_proj_];
        [enc setBuffer:buf_swiglu_out_ offset:0 atIndex:0];
        [enc setBuffer:W_down offset:0 atIndex:1];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&N_mlp length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];

        MTLSize tg_size = MTLSizeMake(32, 1, 1);
        MTLSize grid_size = MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    }

    // 7. Second Residual Add: X_out + buf_proj_out -> X_out
    if (pso_residual_add_) {
        [enc setComputePipelineState:pso_residual_add_];
        [enc setBuffer:X_out offset:0 atIndex:0];
        [enc setBuffer:buf_proj_out_ offset:0 atIndex:1];
        [enc setBuffer:X_out offset:0 atIndex:2];
        uint32_t total = M * K;
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:3];
        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }

    [enc endEncoding];
}

} // namespace metal_llm
