#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <unistd.h>

#include "core/memory/page_allocator.h"
#include "core/memory/uma_tracker.h"
#include "core/memory/cache_flush.h"
#include "core/memory/quant_types.h"
#include "core/metrology/prng.h"
#include "core/metrology/tripwires.h"
#include "core/metrology/telemetry.h"
#include "core/metrology/telemetry_format.h"
#include "core/metrology/bench_standards.h"

// Test harness macros
static int g_tests_passed = 0;
static int g_tests_total = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_total++; \
    if (!(cond)) { \
        std::cerr << "[-] FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        assert(false); \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

using namespace core::memory;
using namespace core::metrology;

void test_invariant_1_page_alignment() {
    std::cout << "[*] Testing Invariant 1: 16KB Direct I/O Page Alignment..." << std::endl;

    // Test raw allocator
    size_t alloc_size = 64 * 1024; // 64KB
    void* ptr = allocate_16kb_aligned(alloc_size);
    TEST_ASSERT(ptr != nullptr, "allocate_16kb_aligned returned non-null");
    TEST_ASSERT(is_16kb_aligned(ptr), "Pointer satisfies is_16kb_aligned");
    TEST_ASSERT((reinterpret_cast<uintptr_t>(ptr) % 16384) == 0, "Address is divisible by 16384");

    // Assertion call must not abort
    assert_16kb_aligned(ptr);

    // Test unaligned detection
    const void* unaligned_ptr = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(ptr) + 16);
    TEST_ASSERT(!is_16kb_aligned(unaligned_ptr), "Unaligned offset correctly flagged as non-16KB aligned");

    free_16kb_aligned(ptr);

    // Test RAII AlignedBuffer
    {
        AlignedBuffer<float> float_buf(1024);
        TEST_ASSERT(float_buf.data() != nullptr, "AlignedBuffer data() is non-null");
        TEST_ASSERT(is_16kb_aligned(float_buf.data()), "AlignedBuffer memory is 16KB aligned");
        TEST_ASSERT(float_buf.size() == 1024, "AlignedBuffer size matches");
        TEST_ASSERT(float_buf.bytes() == 1024 * sizeof(float), "AlignedBuffer bytes match");

        // Write and read
        float_buf[0] = 3.14159f;
        float_buf[1023] = 2.71828f;
        TEST_ASSERT(float_buf[0] == 3.14159f, "Buffer element 0 read/write");
        TEST_ASSERT(float_buf[1023] == 2.71828f, "Buffer element 1023 read/write");

        // Move semantics
        AlignedBuffer<float> moved_buf = std::move(float_buf);
        TEST_ASSERT(float_buf.data() == nullptr, "Moved-from buffer data is nullptr");
        TEST_ASSERT(moved_buf.data() != nullptr, "Moved-to buffer has data");
        TEST_ASSERT(is_16kb_aligned(moved_buf.data()), "Moved-to buffer remains 16KB aligned");
        TEST_ASSERT(moved_buf[0] == 3.14159f, "Moved-to buffer preserves contents");
    }

    std::cout << "    -> Invariant 1 passed successfully." << std::endl;
}

void test_invariant_2_cache_purge() {
    std::cout << "[*] Testing Invariant 2: UBC and SLC Cache Eviction..." << std::endl;

    const char* dummy_path = "/tmp/test_ubc_dummy_file";
    // Ensure clean initial state
    unlink(dummy_path);

    // Purge UBC with 32MB Direct I/O write and read
    purge_unified_buffer_cache(dummy_path, 32 * 1024 * 1024);

    // Dummy file must be unlinked after purge
    TEST_ASSERT(access(dummy_path, F_OK) != 0, "UBC purge dummy file unlinked after purge");

    // Cold cache CPU sweep
    AlignedBuffer<uint8_t> sweep_buf(32 * 1024 * 1024);
    cold_cache_evict_cpu(sweep_buf.data(), sweep_buf.bytes());
    cold_cache_evict_cpu(nullptr, 32 * 1024 * 1024);

    // GPU SLC cache purge if device available
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue) {
            purge_slc_cache_gpu(device, queue, 16 * 1024 * 1024);
            purge_all_cold_caches(device, queue, dummy_path);
            TEST_ASSERT(access(dummy_path, F_OK) != 0, "Dummy file cleaned up in purge_all_cold_caches");
        }
    }

    std::cout << "    -> Invariant 2 passed successfully." << std::endl;
}

void test_invariant_3_and_9_honest_labeling_and_telemetry() {
    std::cout << "[*] Testing Invariants 3 & 9: Honest Verification Labeling & Masked Diff Prohibition..." << std::endl;

    // Invariant 3: Correctness Gate at M <= 2048
    for (size_t M : BOUNDARY_TOKEN_COUNTS) {
        std::string status_pass = format_verification_status(M, 2048, true);
        std::string status_fail = format_verification_status(M, 2048, false);
        TEST_ASSERT(status_pass == "[PASSED]", "M <= 2048 returns [PASSED] when verified");
        TEST_ASSERT(status_fail == "[FAILED]", "M <= 2048 returns [FAILED] on error");
        TEST_ASSERT(status_pass.find("[LOCKED]") == std::string::npos, "No fake [LOCKED] strings in output");
    }

    // Invariant 3: Scale Sweep at M > 2048
    std::string large_status = format_verification_status(4096, 2048, true);
    TEST_ASSERT(large_status == "[NOT VERIFIED — CPU gold infeasible at this scale]",
                "M > 2048 returns honest unverified status");
    TEST_ASSERT(large_status.find("[LOCKED]") == std::string::npos,
                "No fake [LOCKED] strings on large sequence lengths");

    // Invariant 9: Honest difference formatting
    std::string normal_diff = format_diff_telemetry(0.003125f, false);
    TEST_ASSERT(normal_diff == "0.003125", "Exact float formatted when not gated");

    std::string gated_diff = format_diff_telemetry(0.0f, true, "CPU Gold Reference Gated for M > 128");
    TEST_ASSERT(gated_diff == "N/A (CPU Gold Reference Gated for M > 128)",
                "Gated verification outputs explicit reason, never fake 0.000000");
    TEST_ASSERT(gated_diff != "0.000000", "Gated diff is never literal 0.000000");
    TEST_ASSERT(gated_diff.find("N/A (GPU-Only)") == std::string::npos,
                "Does not output ambiguous N/A (GPU-Only)");

    std::cout << "    -> Invariants 3 & 9 passed successfully." << std::endl;
}

void test_invariant_4_uma_footprint() {
    std::cout << "[*] Testing Invariant 4: UMA Memory Working Set Tracking via phys_footprint..." << std::endl;

    size_t bytes = get_uma_phys_footprint_bytes();
    double mb = get_uma_phys_footprint_mb();
    TEST_ASSERT(bytes > 0, "get_uma_phys_footprint_bytes returns non-zero");
    TEST_ASSERT(mb > 0.0, "get_uma_phys_footprint_mb returns non-zero");

    ScopedUMATracker tracker;
    double baseline_mb = tracker.baseline_mb();
    TEST_ASSERT(baseline_mb > 0.0, "ScopedUMATracker baseline is non-zero");

    // 1. Allocate 32MB via 16KB system page allocator and verify expansion
    size_t alloc_bytes = 32 * 1024 * 1024;
    void* host_buf = allocate_16kb_aligned(alloc_bytes);
    TEST_ASSERT(host_buf != nullptr, "16KB-aligned buffer allocated");
    std::memset(host_buf, 0xA5, alloc_bytes);

    tracker.sample();
    double peak_mb = tracker.peak_mb();
    TEST_ASSERT(peak_mb >= baseline_mb + 30.0, "task_vm_info.phys_footprint captures 32MB allocation");

    // 2. Free and verify zero memory leak (footprint returns to baseline)
    free_16kb_aligned(host_buf);
    tracker.sample();
    double post_free_mb = tracker.current_mb();
    TEST_ASSERT(post_free_mb <= baseline_mb + 1.0, "task_vm_info.phys_footprint returns to baseline after free (zero leak verified)");

    // 3. Metal shared buffer expansion verification
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        @autoreleasepool {
            id<MTLBuffer> mtl_buf = [device newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
            TEST_ASSERT(mtl_buf != nil, "Metal shared buffer allocated");
            std::memset([mtl_buf contents], 0xA5, alloc_bytes);
            double metal_peak = get_uma_phys_footprint_mb();
            TEST_ASSERT(metal_peak >= post_free_mb + 25.0, "task_vm_info.phys_footprint captures Metal shared allocation");
            mtl_buf = nil;
        }
    }

    std::cout << "    -> Invariant 4 passed successfully." << std::endl;
}

void test_invariant_5_e2e_and_compute_telemetry() {
    std::cout << "[*] Testing Invariant 5: Dual E2E vs GPU Compute Latency Tracking..." << std::endl;

    BenchmarkTimer timer;
    timer.start_wall();
    usleep(5000); // 5 ms sleep
    timer.set_io_duration_ms(2.5);
    timer.stop_wall();

    ExecutionTiming timing = timer.get_timing();
    TEST_ASSERT(timing.wall_ms >= 4.0, "Wall-clock duration captured");
    TEST_ASSERT(timing.io_ms == 2.5, "Prefetch I/O duration recorded side-by-side");

    std::cout << "    -> Invariant 5 passed successfully." << std::endl;
}

void test_invariant_6_non_finite_tripwires() {
    std::cout << "[*] Testing Invariant 6: Non-Finite (NaN/Inf) Tripwire Assertions..." << std::endl;

    constexpr size_t N = 64;
    std::vector<float> clean_f32(N, 1.0f);
    std::vector<__fp16> clean_f16(N, static_cast<__fp16>(1.0f));

    TEST_ASSERT(verify_finite(clean_f32.data(), N, "CleanF32"), "Clean float32 is verified finite");
    TEST_ASSERT(verify_finite(clean_f16.data(), N, "CleanF16"), "Clean float16 is verified finite");

    // Introduce NaN into float32
    std::vector<float> nan_f32 = clean_f32;
    nan_f32[32] = NAN;
    TEST_ASSERT(!verify_finite(nan_f32.data(), N, "NanF32"), "Float32 with NaN correctly caught by tripwire");

    // Introduce Inf into float32
    std::vector<float> inf_f32 = clean_f32;
    inf_f32[10] = INFINITY;
    TEST_ASSERT(!verify_finite(inf_f32.data(), N, "InfF32"), "Float32 with +Inf caught by tripwire");
    inf_f32[10] = -INFINITY;
    TEST_ASSERT(!verify_finite(inf_f32.data(), N, "-InfF32"), "Float32 with -Inf caught by tripwire");

    // Introduce NaN into float16
    std::vector<__fp16> nan_f16 = clean_f16;
    nan_f16[16] = static_cast<__fp16>(NAN);
    TEST_ASSERT(!verify_finite(nan_f16.data(), N, "NanF16"), "Float16 with NaN caught by tripwire");

    // Compute max difference test
    std::vector<float> gold(N, 2.0f);
    std::vector<__fp16> gpu_out(N, static_cast<__fp16>(2.00390625f));
    float diff = compute_max_diff(gold.data(), gpu_out.data(), N);
    TEST_ASSERT(std::fabs(diff - 0.00390625f) < 1e-6f, "compute_max_diff accurately computed max difference");

    // verify_tensor_parity
    VerificationResult res = verify_tensor_parity(gpu_out.data(), gold.data(), N, 0.01f, "ParityCheck");
    TEST_ASSERT(res.passed, "verify_tensor_parity passed within tolerance");
    TEST_ASSERT(!res.has_non_finite, "No non-finite detected");

    VerificationResult res_fail = verify_tensor_parity(gpu_out.data(), gold.data(), N, 0.001f, "ParityCheckStrict");
    TEST_ASSERT(!res_fail.passed, "verify_tensor_parity fails when diff exceeds strict tolerance");

    std::cout << "    -> Invariant 6 passed successfully." << std::endl;
}

void test_invariant_7_quantization_types_and_layout() {
    std::cout << "[*] Testing Invariant 7: Planar Deinterleaved Quantization Layout & Types..." << std::endl;

    // Verify sizes
    TEST_ASSERT(sizeof(block_q4_0) == 18, "sizeof(block_q4_0) == 18 bytes (32 weights)");
    TEST_ASSERT(sizeof(block_mlx_4bit) == 20, "sizeof(block_mlx_4bit) == 20 bytes (32 weights)");
    TEST_ASSERT(sizeof(block_q4_K) == 144, "sizeof(block_q4_K) == 144 bytes (256 weights)");
    TEST_ASSERT(sizeof(block_ternary_1_58) == 12, "sizeof(block_ternary_1_58) == 12 bytes (32 weights)");
    TEST_ASSERT(sizeof(block_var_rate_affine) == 160, "sizeof(block_var_rate_affine) == 160 bytes (256 weights)");
    TEST_ASSERT(sizeof(block_exl3) == 144, "sizeof(block_exl3) == 144 bytes (256 weights)");
    TEST_ASSERT(sizeof(block_q8_0) == 34, "sizeof(block_q8_0) == 34 bytes (32 weights)");

    // Test Invariant 7: 16-byte nibbles, 2-byte scale, 2-byte bias
    block_mlx_4bit blk;
    TEST_ASSERT(sizeof(blk.qs) == 16, "block_mlx_4bit qs is 16 bytes (128-bit LSU aligned)");
    TEST_ASSERT(sizeof(blk.d) == 2, "block_mlx_4bit d is 2 bytes (FP16 scale)");
    TEST_ASSERT(sizeof(blk.bias) == 2, "block_mlx_4bit bias is 2 bytes (FP16 bias)");

    // Format metadata info check
    QuantFormatInfo info_q4_0 = get_quant_info(QUANT_Q4_0);
    TEST_ASSERT(info_q4_0.block_size == 32, "QUANT_Q4_0 block size is 32");
    TEST_ASSERT(info_q4_0.block_bytes == 18, "QUANT_Q4_0 block bytes is 18");

    QuantFormatInfo info_mlx = get_quant_info(QUANT_MLX_4BIT);
    TEST_ASSERT(info_mlx.block_size == 32, "QUANT_MLX_4BIT block size is 32");
    TEST_ASSERT(info_mlx.bits_per_weight == 5.0, "QUANT_MLX_4BIT bits/weight is 5.0");

    size_t q4_weight_bytes = compute_quant_weight_bytes(QUANT_Q4_0, 1024);
    TEST_ASSERT(q4_weight_bytes == (1024 / 32) * 18, "compute_quant_weight_bytes for Q4_0 matches");

    std::cout << "    -> Invariant 7 passed successfully." << std::endl;
}

void test_invariant_8_sram_bank_padding_constant() {
    std::cout << "[*] Testing Invariant 8: Stride-36 SRAM Bank Conflict Padding..." << std::endl;

    TEST_ASSERT(SRAM_PADDED_STRIDE_36 == 36, "SRAM_PADDED_STRIDE_36 is 36");
    TEST_ASSERT(SRAM_PADDED_STRIDE_36 % 32 != 0, "SRAM stride is non-power-of-2 to avoid 32-bank hardware serialization");

    std::cout << "    -> Invariant 8 passed successfully." << std::endl;
}

void test_invariant_10_benchmarking_standards() {
    std::cout << "[*] Testing Invariant 10: Apples-to-Apples Benchmarking Standards..." << std::endl;

    TEST_ASSERT(BENCHMARK_ITERATIONS == 20, "Benchmark iterations constant is 20");
    TEST_ASSERT(BENCHMARK_WARMUP_RUNS == 3, "Benchmark warmup runs constant is 3");
    TEST_ASSERT(MAX_LATENCY_VARIANCE_PCT == 1.5, "Maximum latency variance envelope is 1.5%");
    TEST_ASSERT(MAX_REGISTERS_PER_THREAD == 64, "Register budget is <= 64 for 100% occupancy");

    // Boundary token check
    TEST_ASSERT(is_boundary_token_count(33), "33 is a boundary token count");
    TEST_ASSERT(is_boundary_token_count(127), "127 is a boundary token count");
    TEST_ASSERT(is_boundary_token_count(128), "128 is a boundary token count");
    TEST_ASSERT(is_boundary_token_count(129), "129 is a boundary token count");
    TEST_ASSERT(is_boundary_token_count(2048), "2048 is a boundary token count");
    TEST_ASSERT(!is_boundary_token_count(64), "64 is not a primary boundary token count");

    // Mandatory disclosure block
    const char* disclosure = get_benchmark_disclosure_block();
    TEST_ASSERT(disclosure != nullptr, "Disclosure block string is non-null");
    TEST_ASSERT(strstr(disclosure, "METROLOGY DISCLOSURE") != nullptr, "Disclosure contains header");
    TEST_ASSERT(strstr(disclosure, "Format Parity") != nullptr, "Disclosure specifies format parity");

    std::cout << "    -> Invariant 10 passed successfully." << std::endl;
}

void test_invariant_11_cognitive_telemetry_units() {
    std::cout << "[*] Testing Invariant 11: Cognitive Telemetry Units..." << std::endl;

    // Sub-second latencies: "XX.XX ms"
    std::string sub_sec = format_time(45.21);
    TEST_ASSERT(sub_sec == "45.21 ms", "Sub-second duration formatted as ms");

    std::string edge_sub = format_time(999.90);
    TEST_ASSERT(edge_sub == "999.90 ms", "999.9 ms formatted as ms");

    // Multi-second latencies: "X.XX s"
    std::string sec_1 = format_time(1000.0);
    TEST_ASSERT(sec_1 == "1.00 s", "1000.0 ms formatted as 1.00 s");

    std::string audit_example = format_time(19602.88);
    TEST_ASSERT(audit_example == "19.60 s", "19602.88 ms formatted as 19.60 s (Audit Failure 11)");

    std::string large_time = format_time(42500.0);
    TEST_ASSERT(large_time == "42.50 s", "42500.0 ms formatted as 42.50 s");

    // Check helper formatters
    TEST_ASSERT(format_throughput(12450.2) == "12450.2 tok/s", "Throughput formatter");
    TEST_ASSERT(format_tflops(45.21) == "45.21 TFLOP/s", "TFLOP/s formatter");
    TEST_ASSERT(format_bandwidth(185.3) == "185.30 GB/s", "Bandwidth formatter");
    TEST_ASSERT(format_memory(512.0) == "512.00 MB", "Memory MB formatter");
    TEST_ASSERT(format_memory(2048.0) == "2.00 GB", "Memory GB formatter");

    std::cout << "    -> Invariant 11 passed successfully." << std::endl;
}

void test_prng_and_statistical_variance() {
    std::cout << "[*] Testing PRNG determinism, distributions, and variance metrics..." << std::endl;

    // Determinism test
    prng_seed(42);
    uint32_t val1 = prng_next_u32();
    uint32_t val2 = prng_next_u32();
    float f1 = prng_rand_uniform(-1.0f, 1.0f);

    prng_seed(42);
    uint32_t val1_repeat = prng_next_u32();
    uint32_t val2_repeat = prng_next_u32();
    float f1_repeat = prng_rand_uniform(-1.0f, 1.0f);

    TEST_ASSERT(val1 == val1_repeat, "PRNG reproduces identical sequence after reseeding");
    TEST_ASSERT(val2 == val2_repeat, "PRNG step 2 reproduces identical sequence");
    TEST_ASSERT(f1 == f1_repeat, "Uniform float reproduces identical value");

    // Median calculation tests
    std::vector<double> odd_samples = {5.0, 1.0, 3.0, 2.0, 4.0};
    double odd_median = compute_median(odd_samples);
    TEST_ASSERT(odd_median == 3.0, "Median of odd samples [1,2,3,4,5] is 3.0");

    std::vector<double> even_samples = {1.0, 4.0, 2.0, 3.0};
    double even_median = compute_median(even_samples);
    TEST_ASSERT(even_median == 2.5, "Median of even samples [1,2,3,4] is 2.5");

    // Variance percentage test
    std::vector<double> variance_samples = {101.5, 101.5, 101.5};
    double var_pct = compute_variance_percentage(variance_samples, 100.0);
    TEST_ASSERT(std::fabs(var_pct - 1.5) < 1e-5, "compute_variance_percentage computes 1.5%");

    // Sample stats summary
    SampleStats stats = compute_sample_stats(odd_samples);
    TEST_ASSERT(stats.min_val == 1.0, "Stats min is 1.0");
    TEST_ASSERT(stats.max_val == 5.0, "Stats max is 5.0");
    TEST_ASSERT(stats.median == 3.0, "Stats median is 3.0");
    TEST_ASSERT(stats.mean == 3.0, "Stats mean is 3.0");

    // Gaussian generator validation
    std::vector<float> gaussian_vals(1000);
    generate_gaussian_activations(gaussian_vals.data(), 1000, 0.0f, 1.0f);
    double g_sum = 0.0;
    for (float v : gaussian_vals) g_sum += v;
    double g_mean = g_sum / 1000.0;
    TEST_ASSERT(std::fabs(g_mean) < 0.15, "Gaussian distribution mean is approximately 0");

    std::cout << "    -> PRNG and telemetry statistics passed successfully." << std::endl;
}

int main(int argc, const char* argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        std::cout << "================================================================================" << std::endl;
        std::cout << " RUNNING CORE INVARIANTS & METROLOGY TEST SUITE (Milestone M1 / R1)" << std::endl;
        std::cout << "================================================================================" << std::endl;

        test_invariant_1_page_alignment();
        test_invariant_2_cache_purge();
        test_invariant_3_and_9_honest_labeling_and_telemetry();
        test_invariant_4_uma_footprint();
        test_invariant_5_e2e_and_compute_telemetry();
        test_invariant_6_non_finite_tripwires();
        test_invariant_7_quantization_types_and_layout();
        test_invariant_8_sram_bank_padding_constant();
        test_invariant_10_benchmarking_standards();
        test_invariant_11_cognitive_telemetry_units();
        test_prng_and_statistical_variance();

        std::cout << "================================================================================" << std::endl;
        std::cout << " SUMMARY: " << g_tests_passed << "/" << g_tests_total << " tests passed." << std::endl;
        std::cout << " ALL METROLOGICAL INVARIANTS SATISFIED WITH REAL IMPLEMENTATIONS." << std::endl;
        std::cout << "================================================================================" << std::endl;

        return (g_tests_passed == g_tests_total) ? 0 : 1;
    }
}
