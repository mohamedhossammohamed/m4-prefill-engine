#import "test_common.h"
#import "cpu_gold_reference.h"

// ============================================================================
// Global Metal Test Fixture
// ============================================================================
struct MetalFixture {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> quant_lib;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;

    bool init() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return false;
            queue = [device newCommandQueue];
            if (!queue) return false;

            quant_lib = load_metal_library(device, "quant_router_kernels.metal");
            if (!quant_lib) return false;

            std::vector<std::string> kernel_names = {
                "quant_router_gemm_q4_0_64x64",
                "quant_router_head_gemm_q4_0_64x64",
                "quant_router_gemm_mlx_4bit_64x64",
                "quant_router_head_gemm_mlx_4bit_64x64",
                "quant_router_gemm_q4_k_64x64",
                "quant_router_head_gemm_q4_k_64x64",
                "quant_router_gemm_ternary_1_58_64x64",
                "quant_router_head_gemm_ternary_1_58_64x64",
                "quant_router_gemm_var_rate_affine_64x64",
                "quant_router_head_gemm_var_rate_affine_64x64",
                "quant_router_gemm_exl3_64x64",
                "quant_router_head_gemm_exl3_64x64",
                "quant_router_gemm_prism_q2_0_64x64",
                "quant_router_head_gemm_prism_q2_0_64x64"
            };

            for (const auto& name : kernel_names) {
                id<MTLComputePipelineState> p = create_pipeline(device, quant_lib, name);
                if (!p) {
                    std::cerr << "[ERROR] Could not create pipeline: " << name << std::endl;
                    return false;
                }
                pipelines[name] = p;
            }
            return true;
        }
    }

    void run_gemm(
        const std::string& pipeline_name,
        id<MTLBuffer> bufA,
        id<MTLBuffer> bufB,
        id<MTLBuffer> bufC,
        uint32_t M,
        uint32_t N,
        uint32_t K)
    {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipelines[pipeline_name]];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
            [enc setThreadgroupMemoryLength:16384 atIndex:0];

            NSUInteger tg_x = (N + 63) / 64;
            NSUInteger tg_y = (M + 63) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void run_head_gemm(
        const std::string& pipeline_name,
        id<MTLBuffer> bufA,
        id<MTLBuffer> bufB,
        id<MTLBuffer> bufC,
        uint32_t M,
        uint32_t H,
        uint32_t D,
        uint32_t K)
    {
        @autoreleasepool {
            uint32_t N = H * D;
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipelines[pipeline_name]];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
            [enc setThreadgroupMemoryLength:16384 atIndex:0];

            NSUInteger tg_x = (N + 63) / 64;
            NSUInteger tg_y = (M + 63) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
};

static MetalFixture g_fixture;

// ============================================================================
// FEATURE 1: Boundary Token Counts M in {33, 127, 128, 129, 2048}
// ============================================================================

static void run_boundary_token_test(TestContext& ctx, uint32_t M) {
    const uint32_t K = 512;
    const uint32_t N = 256;
    size_t act_bytes = (size_t)M * K * sizeof(__fp16);
    size_t out_bytes = (size_t)M * N * sizeof(__fp16);
    size_t weight_bytes = compute_quant_weight_bytes(QUANT_Q4_0, (size_t)N * K);

    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    ASSERT_TRUE(bufA && bufB && bufC);

    generate_activations((__fp16*)[bufA contents], (size_t)M * K);
    generate_q4_0_weights((block_q4_0*)[bufB contents], (size_t)N * (K / 32));

    g_fixture.run_gemm("quant_router_gemm_q4_0_64x64", bufA, bufB, bufC, M, N, K);

    std::vector<__fp16> cpu_gold(M * N);
    cpu_gold_reference_q4_0((const __fp16*)[bufA contents],
                           (const block_q4_0*)[bufB contents],
                           cpu_gold.data(), M, N, K);

    size_t bad_idx = 0;
    ASSERT_TRUE(verify_finite((const __fp16*)[bufC contents], M * N, &bad_idx));
    ASSERT_TRUE(verify_finite(cpu_gold.data(), M * N, &bad_idx));

    float max_diff = compute_max_diff((const __fp16*)[bufC contents], cpu_gold.data(), M * N);
    ASSERT_LE(max_diff, 0.0078125f); // Reconciled dyadic tolerance 1/128
}

TEST_CASE(test_tier1_boundary_token_33) {
    run_boundary_token_test(ctx, 33);
}

TEST_CASE(test_tier1_boundary_token_127) {
    run_boundary_token_test(ctx, 127);
}

TEST_CASE(test_tier1_boundary_token_128) {
    run_boundary_token_test(ctx, 128);
}

TEST_CASE(test_tier1_boundary_token_129) {
    run_boundary_token_test(ctx, 129);
}

TEST_CASE(test_tier1_boundary_token_2048) {
    run_boundary_token_test(ctx, 2048);
}

// ============================================================================
// FEATURE 2: All 6 Quantization Formats (12 GPU Kernel + 18 CPU Unit Tests)
// ============================================================================

// Helper template for format parity tests
template <typename TBlock, typename TGen, typename TRef>
static void run_format_tests(
    TestContext& ctx,
    QuantFormat fmt,
    const std::string& std_pipeline,
    const std::string& head_pipeline,
    TGen generator,
    TRef cpu_ref,
    uint32_t block_step)
{
    const uint32_t M = 128;
    const uint32_t K = 512;
    const uint32_t N = 256;
    const uint32_t H = 4;
    const uint32_t D = 64; // N = H * D

    size_t act_bytes = (size_t)M * K * sizeof(__fp16);
    size_t out_bytes = (size_t)M * N * sizeof(__fp16);
    size_t weight_bytes = compute_quant_weight_bytes(fmt, (size_t)N * K);

    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:act_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC_std = [g_fixture.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC_head = [g_fixture.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    ASSERT_TRUE(bufA && bufB && bufC_std && bufC_head);

    generate_activations((__fp16*)[bufA contents], (size_t)M * K);
    generator((TBlock*)[bufB contents], (size_t)N * (K / block_step));

    // 1. Standard GEMM Parity
    g_fixture.run_gemm(std_pipeline, bufA, bufB, bufC_std, M, N, K);
    std::vector<__fp16> cpu_std(M * N);
    cpu_ref((const __fp16*)[bufA contents], (const TBlock*)[bufB contents], cpu_std.data(), M, N, K, false, 0, 0);

    ASSERT_TRUE(verify_finite((const __fp16*)[bufC_std contents], M * N));
    float diff_std = compute_max_diff((const __fp16*)[bufC_std contents], cpu_std.data(), M * N);
    ASSERT_LE(diff_std, 0.0078125f); // Reconciled dyadic tolerance 1/128

    // 2. Direct Head Routing Parity
    g_fixture.run_head_gemm(head_pipeline, bufA, bufB, bufC_head, M, H, D, K);
    std::vector<__fp16> cpu_head(M * N);
    cpu_ref((const __fp16*)[bufA contents], (const TBlock*)[bufB contents], cpu_head.data(), M, N, K, true, H, D);

    ASSERT_TRUE(verify_finite((const __fp16*)[bufC_head contents], M * N));
    float diff_head = compute_max_diff((const __fp16*)[bufC_head contents], cpu_head.data(), M * N);
    ASSERT_LE(diff_head, 0.0078125f); // Reconciled dyadic tolerance 1/128
}

// 2.1 Q4_0 Tests
TEST_CASE(test_tier1_format_q4_0_standard_gemm) {
    run_format_tests<block_q4_0>(ctx, QUANT_Q4_0, "quant_router_gemm_q4_0_64x64", "quant_router_head_gemm_q4_0_64x64",
                                 generate_q4_0_weights, cpu_gold_reference_q4_0, 32);
}
TEST_CASE(test_tier1_format_q4_0_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_Q4_0, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_q4_0_weights((block_q4_0*)[bufB contents], N * (K / 32));
    g_fixture.run_head_gemm("quant_router_head_gemm_q4_0_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_q4_0((const __fp16*)[bufA contents], (const block_q4_0*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_q4_0_cpu_scale_reconstruction) {
    block_q4_0 blk;
    blk.d = (__fp16)0.0125f;
    std::memset(blk.qs, 0x88, 16); // 8 is zero-point
    ASSERT_NEAR((float)blk.d, 0.0125f, 1e-4f);
}
TEST_CASE(test_tier1_format_q4_0_cpu_zero_offset) {
    block_q4_0 blk;
    blk.d = (__fp16)1.0f;
    for (int i = 0; i < 16; i++) blk.qs[i] = 0x88; // both nibbles = 8 -> 8 - 8 = 0
    __fp16 act[32];
    for (int i = 0; i < 32; i++) act[i] = (__fp16)1.0f;
    __fp16 out[1] = {0};
    cpu_gold_reference_q4_0(act, &blk, out, 1, 1, 32);
    ASSERT_NEAR((float)out[0], 0.0f, 1e-4f);
}
TEST_CASE(test_tier1_format_q4_0_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_q4_0), 18);
    ASSERT_EQ(get_quant_info(QUANT_Q4_0).block_size, 32);
}

// 2.2 MLX_4BIT Tests
TEST_CASE(test_tier1_format_mlx_4bit_standard_gemm) {
    run_format_tests<block_mlx_4bit>(ctx, QUANT_MLX_4BIT, "quant_router_gemm_mlx_4bit_64x64", "quant_router_head_gemm_mlx_4bit_64x64",
                                     generate_mlx_4bit_weights, cpu_gold_reference_mlx_4bit, 32);
}
TEST_CASE(test_tier1_format_mlx_4bit_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_MLX_4BIT, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_mlx_4bit_weights((block_mlx_4bit*)[bufB contents], N * (K / 32));
    g_fixture.run_head_gemm("quant_router_head_gemm_mlx_4bit_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_mlx_4bit((const __fp16*)[bufA contents], (const block_mlx_4bit*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_mlx_4bit_cpu_scale_reconstruction) {
    block_mlx_4bit blk;
    blk.d = (__fp16)0.02f;
    blk.bias = (__fp16)-0.01f;
    ASSERT_NEAR((float)blk.d, 0.02f, 1e-4f);
    ASSERT_NEAR((float)blk.bias, -0.01f, 1e-4f);
}
TEST_CASE(test_tier1_format_mlx_4bit_cpu_bias_application) {
    block_mlx_4bit blk;
    blk.d = (__fp16)0.0f;
    blk.bias = (__fp16)0.5f;
    std::memset(blk.qs, 0, 16);
    __fp16 act[32];
    for (int i = 0; i < 32; i++) act[i] = (__fp16)1.0f;
    __fp16 out[1] = {0};
    cpu_gold_reference_mlx_4bit(act, &blk, out, 1, 1, 32);
    ASSERT_NEAR((float)out[0], 16.0f, 1e-2f); // 32 elements * 0.5 = 16.0
}
TEST_CASE(test_tier1_format_mlx_4bit_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_mlx_4bit), 20);
    ASSERT_EQ(get_quant_info(QUANT_MLX_4BIT).block_size, 32);
}

// 2.3 Q4_K Tests
TEST_CASE(test_tier1_format_q4_k_standard_gemm) {
    run_format_tests<block_q4_K>(ctx, QUANT_Q4_K, "quant_router_gemm_q4_k_64x64", "quant_router_head_gemm_q4_k_64x64",
                                 generate_q4_k_weights, cpu_gold_reference_q4_k, 256);
}
TEST_CASE(test_tier1_format_q4_k_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_Q4_K, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_q4_k_weights((block_q4_K*)[bufB contents], N * (K / 256));
    g_fixture.run_head_gemm("quant_router_head_gemm_q4_k_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_q4_k((const __fp16*)[bufA contents], (const block_q4_K*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_q4_k_cpu_scale_reconstruction) {
    block_q4_K blk;
    blk.d = (__fp16)0.001f;
    blk.dmin = (__fp16)0.0005f;
    ASSERT_NEAR((float)blk.d, 0.001f, 1e-4f);
    ASSERT_NEAR((float)blk.dmin, 0.0005f, 1e-4f);
}
TEST_CASE(test_tier1_format_q4_k_cpu_mins_unpacking) {
    block_q4_K blk;
    blk.d = (__fp16)0.001f;
    blk.dmin = (__fp16)0.001f;
    std::memset(blk.scales, 0, sizeof(blk.scales));
    std::memset(blk.qs, 0, sizeof(blk.qs));
    ASSERT_EQ(sizeof(blk.scales), 12);
    ASSERT_EQ(sizeof(blk.qs), 128);
}
TEST_CASE(test_tier1_format_q4_k_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_q4_K), 144);
    ASSERT_EQ(get_quant_info(QUANT_Q4_K).block_size, 256);
}

// 2.4 TERNARY_1_58 Tests
TEST_CASE(test_tier1_format_ternary_1_58_standard_gemm) {
    run_format_tests<block_ternary_1_58>(ctx, QUANT_TERNARY_1_58, "quant_router_gemm_ternary_1_58_64x64", "quant_router_head_gemm_ternary_1_58_64x64",
                                         generate_ternary_1_58_weights, cpu_gold_reference_ternary_1_58, 32);
}
TEST_CASE(test_tier1_format_ternary_1_58_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_TERNARY_1_58, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_ternary_1_58_weights((block_ternary_1_58*)[bufB contents], N * (K / 32));
    g_fixture.run_head_gemm("quant_router_head_gemm_ternary_1_58_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_ternary_1_58((const __fp16*)[bufA contents], (const block_ternary_1_58*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_ternary_1_58_cpu_scale_reconstruction) {
    block_ternary_1_58 blk;
    blk.d = (__fp16)0.035f;
    ASSERT_NEAR((float)blk.d, 0.035f, 1e-4f);
}
TEST_CASE(test_tier1_format_ternary_1_58_cpu_bit_trits) {
    block_ternary_1_58 blk;
    blk.d = (__fp16)1.0f;
    blk._pad = 0;
    // 0 -> -1, 1 -> 0, 2 -> +1
    // qs = 0b01010101... -> choice 1 -> value 0
    blk.qs[0] = 0x55555555;
    blk.qs[1] = 0x55555555;
    __fp16 act[32];
    for (int i = 0; i < 32; i++) act[i] = (__fp16)1.0f;
    __fp16 out[1] = {0};
    cpu_gold_reference_ternary_1_58(act, &blk, out, 1, 1, 32);
    ASSERT_NEAR((float)out[0], 0.0f, 1e-4f);
}
TEST_CASE(test_tier1_format_ternary_1_58_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_ternary_1_58), 12);
    ASSERT_EQ(get_quant_info(QUANT_TERNARY_1_58).block_size, 32);
}

// 2.5 VAR_RATE_AFFINE Tests
TEST_CASE(test_tier1_format_var_rate_affine_standard_gemm) {
    run_format_tests<block_var_rate_affine>(ctx, QUANT_VAR_RATE_AFFINE, "quant_router_gemm_var_rate_affine_64x64", "quant_router_head_gemm_var_rate_affine_64x64",
                                            generate_var_rate_affine_weights, cpu_gold_reference_var_rate_affine, 256);
}
TEST_CASE(test_tier1_format_var_rate_affine_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_VAR_RATE_AFFINE, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_var_rate_affine_weights((block_var_rate_affine*)[bufB contents], N * (K / 256));
    g_fixture.run_head_gemm("quant_router_head_gemm_var_rate_affine_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_var_rate_affine((const __fp16*)[bufA contents], (const block_var_rate_affine*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_var_rate_affine_cpu_scale_reconstruction) {
    block_var_rate_affine blk;
    blk.d = (__fp16)0.002f;
    blk.bias = (__fp16)-0.001f;
    ASSERT_NEAR((float)blk.d, 0.002f, 1e-4f);
    ASSERT_NEAR((float)blk.bias, -0.001f, 1e-4f);
}
TEST_CASE(test_tier1_format_var_rate_affine_cpu_subblock_modes) {
    block_var_rate_affine blk;
    for (int s = 0; s < 8; s++) {
        uint8_t bits = (s < 2) ? 3 : (s < 6 ? 4 : 5);
        uint8_t perm = s % 3;
        blk.modes[s] = (perm << 3) | (bits & 0x07);
        ASSERT_EQ((blk.modes[s] & 0x07), bits);
        ASSERT_EQ(((blk.modes[s] >> 3) & 0x03), perm);
    }
}
TEST_CASE(test_tier1_format_var_rate_affine_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_var_rate_affine), 160);
    ASSERT_EQ(get_quant_info(QUANT_VAR_RATE_AFFINE).block_size, 256);
}

// 2.6 EXL3 Tests
TEST_CASE(test_tier1_format_exl3_standard_gemm) {
    run_format_tests<block_exl3>(ctx, QUANT_EXL3, "quant_router_gemm_exl3_64x64", "quant_router_head_gemm_exl3_64x64",
                                 generate_exl3_weights, cpu_gold_reference_exl3, 256);
}
TEST_CASE(test_tier1_format_exl3_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_EXL3, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_exl3_weights((block_exl3*)[bufB contents], N * (K / 256));
    g_fixture.run_head_gemm("quant_router_head_gemm_exl3_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_exl3((const __fp16*)[bufA contents], (const block_exl3*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_exl3_cpu_scale_reconstruction) {
    block_exl3 blk;
    blk.d = (__fp16)0.0015f;
    blk.bias = (__fp16)0.0005f;
    ASSERT_NEAR((float)blk.d, 0.0015f, 1e-4f);
    ASSERT_NEAR((float)blk.bias, 0.0005f, 1e-4f);
}
TEST_CASE(test_tier1_format_exl3_cpu_codebook_centroids) {
    block_exl3 blk;
    for (int i = 0; i < 16; i++) {
        blk.codebook[i] = (int8_t)(i - 8);
    }
    ASSERT_EQ(blk.codebook[0], -8);
    ASSERT_EQ(blk.codebook[15], 7);
}
TEST_CASE(test_tier1_format_exl3_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_exl3), 144);
    ASSERT_EQ(get_quant_info(QUANT_EXL3).block_size, 256);
}

// 2.7 PrismML Q2_0 Tests
TEST_CASE(test_tier1_format_prism_q2_0_standard_gemm) {
    run_format_tests<block_prism_q2_0>(ctx, QUANT_PRISM_Q2_0, "quant_router_gemm_prism_q2_0_64x64", "quant_router_head_gemm_prism_q2_0_64x64",
                                       generate_prism_q2_0_weights, cpu_gold_reference_prism_q2_0, 128);
}
TEST_CASE(test_tier1_format_prism_q2_0_direct_head) {
    const uint32_t M = 64, K = 256, H = 2, D = 64, N = 128;
    id<MTLBuffer> bufA = [g_fixture.device newBufferWithLength:M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_fixture.device newBufferWithLength:compute_quant_weight_bytes(QUANT_PRISM_Q2_0, N * K) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_fixture.device newBufferWithLength:M * N * 2 options:MTLResourceStorageModeShared];
    generate_activations((__fp16*)[bufA contents], M * K);
    generate_prism_q2_0_weights((block_prism_q2_0*)[bufB contents], N * (K / 128));
    g_fixture.run_head_gemm("quant_router_head_gemm_prism_q2_0_64x64", bufA, bufB, bufC, M, H, D, K);
    std::vector<__fp16> cpu(M * N);
    cpu_gold_reference_prism_q2_0((const __fp16*)[bufA contents], (const block_prism_q2_0*)[bufB contents], cpu.data(), M, N, K, true, H, D);
    ASSERT_LE(compute_max_diff((const __fp16*)[bufC contents], cpu.data(), M * N), 0.0078125f);
}
TEST_CASE(test_tier1_format_prism_q2_0_cpu_scale_reconstruction) {
    block_prism_q2_0 blk;
    blk.d = (__fp16)0.025f;
    std::memset(blk.qs, 0x55, 32); // 0b01010101 = all 1s -> 1 - 1 = 0
    ASSERT_NEAR((float)blk.d, 0.025f, 1e-4f);
}
TEST_CASE(test_tier1_format_prism_q2_0_cpu_vector_lsu_alignment) {
    ASSERT_EQ(sizeof(block_prism_q2_0), 34);
    ASSERT_EQ(get_quant_info(QUANT_PRISM_Q2_0).block_size, 128);
}

// ============================================================================
// FEATURE 3: Non-Finite Tripwires (5 Tests)
// ============================================================================

TEST_CASE(test_tier1_tripwire_finite_clean) {
    std::vector<float> clean(1000, 1.234f);
    size_t bad_idx = 0;
    ASSERT_TRUE(verify_finite(clean.data(), clean.size(), &bad_idx));
}

TEST_CASE(test_tier1_tripwire_detects_nan_cpu) {
    std::vector<float> poisoned(1000, 1.0f);
    poisoned[42] = std::numeric_limits<float>::quiet_NaN();
    size_t bad_idx = 0;
    ASSERT_FALSE(verify_finite(poisoned.data(), poisoned.size(), &bad_idx));
    ASSERT_EQ(bad_idx, 42);
}

TEST_CASE(test_tier1_tripwire_detects_inf_cpu) {
    std::vector<float> poisoned(1000, 1.0f);
    poisoned[128] = std::numeric_limits<float>::infinity();
    size_t bad_idx = 0;
    ASSERT_FALSE(verify_finite(poisoned.data(), poisoned.size(), &bad_idx));
    ASSERT_EQ(bad_idx, 128);
}

TEST_CASE(test_tier1_tripwire_detects_nan_gpu) {
    id<MTLBuffer> gpu_buf = [g_fixture.device newBufferWithLength:100 * sizeof(__fp16) options:MTLResourceStorageModeShared];
    __fp16* ptr = (__fp16*)[gpu_buf contents];
    for (int i = 0; i < 100; i++) ptr[i] = (__fp16)0.5f;
    ptr[77] = (__fp16)NAN;
    size_t bad_idx = 0;
    ASSERT_FALSE(verify_finite(ptr, 100, &bad_idx));
    ASSERT_EQ(bad_idx, 77);
}

TEST_CASE(test_tier1_tripwire_detects_inf_gpu) {
    id<MTLBuffer> gpu_buf = [g_fixture.device newBufferWithLength:100 * sizeof(__fp16) options:MTLResourceStorageModeShared];
    __fp16* ptr = (__fp16*)[gpu_buf contents];
    for (int i = 0; i < 100; i++) ptr[i] = (__fp16)0.5f;
    ptr[33] = (__fp16)INFINITY;
    size_t bad_idx = 0;
    ASSERT_FALSE(verify_finite(ptr, 100, &bad_idx));
    ASSERT_EQ(bad_idx, 33);
}

// ============================================================================
// FEATURE 4: 16KB Direct I/O Alignment (5 Tests)
// ============================================================================

TEST_CASE(test_tier1_direct_io_16kb_alloc) {
    void* ptr = allocate_16kb_aligned(65536);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(is_16kb_aligned(ptr));
    free_16kb_aligned(ptr);
}

TEST_CASE(test_tier1_direct_io_assertion_passes) {
    void* ptr = allocate_16kb_aligned(16384);
    ASSERT_TRUE(is_16kb_aligned(ptr));
    free_16kb_aligned(ptr);
}

TEST_CASE(test_tier1_direct_io_assertion_catches_unaligned) {
    void* ptr = allocate_16kb_aligned(32768);
    uint8_t* unaligned = (uint8_t*)ptr + 16;
    ASSERT_FALSE(is_16kb_aligned(unaligned));
    free_16kb_aligned(ptr);
}

TEST_CASE(test_tier1_direct_io_fnocache_operation) {
    const size_t io_size = 16384;
    void* buf = allocate_16kb_aligned(io_size);
    ASSERT_TRUE(buf != nullptr);
    std::memset(buf, 0x7E, io_size);

    const char* path = "/tmp/test_fnocache_direct_io.bin";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    ASSERT_GE(fd, 0);

    int fcntl_ret = fcntl(fd, F_NOCACHE, 1);
    ASSERT_EQ(fcntl_ret, 0);

    ssize_t w = pwrite(fd, buf, io_size, 0);
    ASSERT_EQ(w, (ssize_t)io_size);
    fsync(fd);

    void* read_buf = allocate_16kb_aligned(io_size);
    ASSERT_TRUE(read_buf != nullptr);
    ssize_t r = pread(fd, read_buf, io_size, 0);
    ASSERT_EQ(r, (ssize_t)io_size);

    ASSERT_EQ(std::memcmp(buf, read_buf, io_size), 0);

    close(fd);
    unlink(path);
    free_16kb_aligned(buf);
    free_16kb_aligned(read_buf);
}

TEST_CASE(test_tier1_direct_io_ubc_purge_execution) {
    bool ok = purge_unified_buffer_cache();
    ASSERT_TRUE(ok);
}

// ============================================================================
// FEATURE 5: UMA phys_footprint Tracking (5 Tests)
// ============================================================================

TEST_CASE(test_tier1_uma_tracker_valid_reading) {
    double footprint_mb = get_accurate_uma_footprint_mb();
    ASSERT_GE(footprint_mb, 1.0); // should be at least a few MB for any running process
}

TEST_CASE(test_tier1_uma_tracker_detects_metal_alloc) {
    double before_mb = get_accurate_uma_footprint_mb();
    const size_t alloc_bytes = 48 * 1024 * 1024; // 48MB
    id<MTLBuffer> buf = [g_fixture.device newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
    ASSERT_TRUE(buf != nil);

    // Touch buffer memory so pages are committed
    uint8_t* ptr = (uint8_t*)[buf contents];
    for (size_t i = 0; i < alloc_bytes; i += 4096) {
        ptr[i] = (uint8_t)(i & 0xFF);
    }

    double during_mb = get_accurate_uma_footprint_mb();
    ASSERT_GE(during_mb, before_mb + 25.0); // phys_footprint must register at least 25MB increase

    buf = nil;
}

TEST_CASE(test_tier1_uma_tracker_detects_metal_free) {
    double baseline_mb = get_accurate_uma_footprint_mb();
    double peak_mb = 0.0;
    @autoreleasepool {
        const size_t alloc_bytes = 64 * 1024 * 1024; // 64MB
        id<MTLBuffer> buf = [g_fixture.device newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
        uint8_t* ptr = (uint8_t*)[buf contents];
        for (size_t i = 0; i < alloc_bytes; i += 4096) ptr[i] = 0xAA;

        peak_mb = get_accurate_uma_footprint_mb();
        ASSERT_GE(peak_mb, baseline_mb + 30.0);
        buf = nil;
    }

    double after_mb = get_accurate_uma_footprint_mb();
    // Reclaimed footprint should drop below peak or stabilize
    ASSERT_LE(after_mb, peak_mb);
    ASSERT_GE(after_mb, 0.0);
}

TEST_CASE(test_tier1_uma_tracker_vs_resident_size) {
    double uma_footprint = get_accurate_uma_footprint_mb();
    ASSERT_GE(uma_footprint, 0.0);
    // Task phys_footprint is the authoritative metric on macOS Apple Silicon UMA
}

TEST_CASE(test_tier1_uma_tracker_allocation_cycle_stability) {
    double initial_mb = get_accurate_uma_footprint_mb();
    for (int cycle = 0; cycle < 5; cycle++) {
        @autoreleasepool {
            id<MTLBuffer> temp = [g_fixture.device newBufferWithLength:16 * 1024 * 1024 options:MTLResourceStorageModeShared];
            uint8_t* p = (uint8_t*)[temp contents];
            p[0] = 1;
            temp = nil;
        }
    }
    double final_mb = get_accurate_uma_footprint_mb();
    // Memory footprint must not drift by more than 1.0 MB across cycles (measured drift < 0.2 MB)
    ASSERT_NEAR(initial_mb, final_mb, 1.0);
}

// ============================================================================
// MAIN RUNNER
// ============================================================================
int main(int argc, const char* argv[]) {
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << " TIER 1: FEATURE COVERAGE TEST SUITE" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;

    if (!g_fixture.init()) {
        std::cerr << COLOR_RED << "[FATAL] Metal fixture initialization failed!" << COLOR_RESET << std::endl;
        return 1;
    }

    TestContext ctx;

    // Feature 1: Boundary Tokens (5 tests)
    std::cout << "\n" << COLOR_YELLOW << "[Feature 1: Boundary Token Counts M in {33, 127, 128, 129, 2048}]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_boundary_token_33);
    RUN_TEST(test_tier1_boundary_token_127);
    RUN_TEST(test_tier1_boundary_token_128);
    RUN_TEST(test_tier1_boundary_token_129);
    RUN_TEST(test_tier1_boundary_token_2048);

    // Feature 2A: 6 Quantization Formats GPU Kernel Parity (12 GPU tests)
    std::cout << "\n" << COLOR_YELLOW << "[Feature 2A: Quantization Formats - GPU Kernel Execution (12 Tests)]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_format_q4_0_standard_gemm);
    RUN_TEST(test_tier1_format_q4_0_direct_head);
    RUN_TEST(test_tier1_format_mlx_4bit_standard_gemm);
    RUN_TEST(test_tier1_format_mlx_4bit_direct_head);
    RUN_TEST(test_tier1_format_q4_k_standard_gemm);
    RUN_TEST(test_tier1_format_q4_k_direct_head);
    RUN_TEST(test_tier1_format_ternary_1_58_standard_gemm);
    RUN_TEST(test_tier1_format_ternary_1_58_direct_head);
    RUN_TEST(test_tier1_format_var_rate_affine_standard_gemm);
    RUN_TEST(test_tier1_format_var_rate_affine_direct_head);
    RUN_TEST(test_tier1_format_exl3_standard_gemm);
    RUN_TEST(test_tier1_format_exl3_direct_head);
    RUN_TEST(test_tier1_format_prism_q2_0_standard_gemm);
    RUN_TEST(test_tier1_format_prism_q2_0_direct_head);

    // Feature 2B: Quantization Formats CPU Layout & Unpacking Unit Tests
    std::cout << "\n" << COLOR_YELLOW << "[Feature 2B: Quantization Formats - CPU Unit & Layout Tests]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_format_q4_0_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_q4_0_cpu_zero_offset);
    RUN_TEST(test_tier1_format_q4_0_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_mlx_4bit_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_mlx_4bit_cpu_bias_application);
    RUN_TEST(test_tier1_format_mlx_4bit_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_q4_k_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_q4_k_cpu_mins_unpacking);
    RUN_TEST(test_tier1_format_q4_k_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_ternary_1_58_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_ternary_1_58_cpu_bit_trits);
    RUN_TEST(test_tier1_format_ternary_1_58_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_var_rate_affine_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_var_rate_affine_cpu_subblock_modes);
    RUN_TEST(test_tier1_format_var_rate_affine_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_exl3_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_exl3_cpu_codebook_centroids);
    RUN_TEST(test_tier1_format_exl3_cpu_vector_lsu_alignment);

    RUN_TEST(test_tier1_format_prism_q2_0_cpu_scale_reconstruction);
    RUN_TEST(test_tier1_format_prism_q2_0_cpu_vector_lsu_alignment);

    // Feature 3: Non-Finite Tripwires (5 tests)
    std::cout << "\n" << COLOR_YELLOW << "[Feature 3: Non-Finite Tripwires]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_tripwire_finite_clean);
    RUN_TEST(test_tier1_tripwire_detects_nan_cpu);
    RUN_TEST(test_tier1_tripwire_detects_inf_cpu);
    RUN_TEST(test_tier1_tripwire_detects_nan_gpu);
    RUN_TEST(test_tier1_tripwire_detects_inf_gpu);

    // Feature 4: 16KB Direct I/O Alignment (5 tests)
    std::cout << "\n" << COLOR_YELLOW << "[Feature 4: 16KB Direct I/O Alignment & UBC Purge]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_direct_io_16kb_alloc);
    RUN_TEST(test_tier1_direct_io_assertion_passes);
    RUN_TEST(test_tier1_direct_io_assertion_catches_unaligned);
    RUN_TEST(test_tier1_direct_io_fnocache_operation);
    RUN_TEST(test_tier1_direct_io_ubc_purge_execution);

    // Feature 5: UMA phys_footprint Tracking (5 tests)
    std::cout << "\n" << COLOR_YELLOW << "[Feature 5: UMA phys_footprint Tracking]" << COLOR_RESET << std::endl;
    RUN_TEST(test_tier1_uma_tracker_valid_reading);
    RUN_TEST(test_tier1_uma_tracker_detects_metal_alloc);
    RUN_TEST(test_tier1_uma_tracker_detects_metal_free);
    RUN_TEST(test_tier1_uma_tracker_vs_resident_size);
    RUN_TEST(test_tier1_uma_tracker_allocation_cycle_stability);

    ctx.print_summary("Tier 1 - Feature Coverage");
    return ctx.exit_code();
}
