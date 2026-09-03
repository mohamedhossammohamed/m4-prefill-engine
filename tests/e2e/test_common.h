#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/task_info.h>
#include <dispatch/dispatch.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "../../quant_router.h"

// ============================================================================
// ANSI Color Codes
// ============================================================================
#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_CYAN    "\033[36m"

// ============================================================================
// Test Framework Harness
// ============================================================================
struct TestContext {
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    std::string current_test_name;
    bool current_test_failed = false;
    std::string failure_reason;

    void start_test(const std::string& name) {
        total_tests++;
        current_test_name = name;
        current_test_failed = false;
        failure_reason.clear();
        std::cout << "  " << COLOR_CYAN << "RUN " << COLOR_RESET << name << "... " << std::flush;
    }

    void record_pass() {
        if (!current_test_failed) {
            passed_tests++;
            std::cout << COLOR_GREEN << "PASS" << COLOR_RESET << std::endl;
        }
    }

    void record_failure(const std::string& reason, const char* file, int line) {
        current_test_failed = true;
        failed_tests++;
        std::ostringstream ss;
        ss << reason << " (at " << file << ":" << line << ")";
        failure_reason = ss.str();
        std::cout << COLOR_RED << "FAIL: " << failure_reason << COLOR_RESET << std::endl;
    }

    int exit_code() const {
        return (failed_tests == 0) ? 0 : 1;
    }

    void print_summary(const std::string& tier_name) const {
        std::cout << "\n" << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
        std::cout << COLOR_BOLD << " SUMMARY: " << tier_name << COLOR_RESET << std::endl;
        std::cout << "  Total Tests:  " << total_tests << std::endl;
        std::cout << "  Passed Tests: " << COLOR_GREEN << passed_tests << COLOR_RESET << std::endl;
        std::cout << "  Failed Tests: " << (failed_tests > 0 ? COLOR_RED : COLOR_GREEN) << failed_tests << COLOR_RESET << std::endl;
        std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    }
};

#define TEST_CASE(name) void name(TestContext& ctx)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        ctx.record_failure("ASSERT_TRUE failed: (" #cond ") is false", __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        ctx.record_failure("ASSERT_FALSE failed: (" #cond ") is true", __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::ostringstream _ss; \
        _ss << "ASSERT_EQ failed: " #a " (" << (a) << ") != " #b " (" << (b) << ")"; \
        ctx.record_failure(_ss.str(), __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define ASSERT_NEAR(a, b, eps) do { \
    double _diff = std::fabs((double)(a) - (double)(b)); \
    if (_diff > (eps)) { \
        std::ostringstream _ss; \
        _ss << "ASSERT_NEAR failed: |" #a " - " #b "| = " << _diff << " > " << (eps) \
            << " (" #a "=" << (a) << ", " #b "=" << (b) << ")"; \
        ctx.record_failure(_ss.str(), __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define ASSERT_LE(a, b) do { \
    if ((a) > (b)) { \
        std::ostringstream _ss; \
        _ss << "ASSERT_LE failed: " #a " (" << (a) << ") > " #b " (" << (b) << ")"; \
        ctx.record_failure(_ss.str(), __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define ASSERT_GE(a, b) do { \
    if ((a) < (b)) { \
        std::ostringstream _ss; \
        _ss << "ASSERT_GE failed: " #a " (" << (a) << ") < " #b " (" << (b) << ")"; \
        ctx.record_failure(_ss.str(), __FILE__, __LINE__); \
        return; \
    } \
} while (0)

#define RUN_TEST(func) do { \
    ctx.start_test(#func); \
    func(ctx); \
    ctx.record_pass(); \
} while (0)

// ============================================================================
// Metrology & Invariant 1: 16KB Direct I/O Page Alignment
// ============================================================================
inline bool is_16kb_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16384 == 0);
}

inline void* allocate_16kb_aligned(size_t bytes) {
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, 16384, bytes);
    if (ret != 0) return nullptr;
    return ptr;
}

inline void free_16kb_aligned(void* ptr) {
    free(ptr);
}

// ============================================================================
// Metrology & Invariant 2: Unified Buffer Cache (UBC) Purge
// ============================================================================
inline bool purge_unified_buffer_cache(const char* dummy_path = "/tmp/ubc_purge_dummy_test") {
    const size_t purge_size = 32 * 1024 * 1024; // 32MB
    void* dummy = allocate_16kb_aligned(purge_size);
    if (!dummy) return false;
    std::memset(dummy, 0x5A, purge_size);

    int fd = open(dummy_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        free_16kb_aligned(dummy);
        return false;
    }
    fcntl(fd, F_NOCACHE, 1);
    ssize_t written = pwrite(fd, dummy, purge_size, 0);
    fsync(fd);
    ssize_t read_bytes = pread(fd, dummy, purge_size, 0);
    close(fd);
    unlink(dummy_path);
    free_16kb_aligned(dummy);

    return (written == (ssize_t)purge_size && read_bytes == (ssize_t)purge_size);
}

// ============================================================================
// Metrology & Invariant 4: Hardware-Accurate UMA Memory Working Set Tracking
// ============================================================================
inline double get_accurate_uma_footprint_mb() {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count);
    if (kr != KERN_SUCCESS) return 0.0;
    return (double)vm_info.phys_footprint / (1024.0 * 1024.0);
}

inline double get_process_rss_mb() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (double)info.resident_size / (1024.0 * 1024.0);
    }
    return 0.0;
}

// ============================================================================
// Metrology & Invariant 6: Non-Finite (NaN/Inf) Tripwires
// ============================================================================
inline bool verify_finite(const float* data, size_t count, size_t* out_bad_idx = nullptr) {
    for (size_t i = 0; i < count; i++) {
        float val = data[i];
        if (std::isnan(val) || std::isinf(val)) {
            if (out_bad_idx) *out_bad_idx = i;
            return false;
        }
    }
    return true;
}

inline bool verify_finite(const __fp16* data, size_t count, size_t* out_bad_idx = nullptr) {
    for (size_t i = 0; i < count; i++) {
        float val = (float)data[i];
        if (std::isnan(val) || std::isinf(val)) {
            if (out_bad_idx) *out_bad_idx = i;
            return false;
        }
    }
    return true;
}

inline float compute_max_diff(const __fp16* a, const __fp16* b, size_t count, size_t* max_idx = nullptr) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float va = (float)a[i];
        float vb = (float)b[i];
        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
            return 999999.0f; // tripwire on NaN/Inf in diff loop
        }
        float diff = std::fabs(va - vb);
        if (diff > max_diff) {
            max_diff = diff;
            if (max_idx) *max_idx = i;
        }
    }
    return max_diff;
}

// ============================================================================
// Synthetic PRNG Generators
// ============================================================================
static uint32_t g_prng_state = 1337;

inline void prng_seed(uint32_t seed) {
    g_prng_state = seed;
}

inline uint32_t prng_next_u32() {
    g_prng_state = g_prng_state * 1664525u + 1013904223u;
    return g_prng_state;
}

inline float prng_rand_uniform(float min_val = -1.0f, float max_val = 1.0f) {
    float u = (float)prng_next_u32() / (float)0xFFFFFFFF;
    return min_val + u * (max_val - min_val);
}

inline void generate_activations(__fp16* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float u1 = std::max(1e-6f, (float)prng_next_u32() / 4294967295.0f);
        float u2 = (float)prng_next_u32() / 4294967295.0f;
        float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
        data[i] = (__fp16)(z0 * 0.35f);
    }
}

// Procedural weight generators for all 6 formats
inline void generate_q4_0_weights(block_q4_0* blocks, size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = (__fp16)(prng_rand_uniform(0.002f, 0.04f));
        for (int i = 0; i < 16; i++) {
            uint8_t low = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            uint8_t high = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

inline void generate_mlx_4bit_weights(block_mlx_4bit* blocks, size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = (__fp16)(prng_rand_uniform(0.001f, 0.03f));
        blocks[b].bias = (__fp16)(prng_rand_uniform(-0.05f, 0.05f));
        for (int i = 0; i < 16; i++) {
            uint8_t low = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            uint8_t high = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

inline void generate_q4_k_weights(block_q4_K* blocks, size_t num_superblocks) {
    for (size_t sb = 0; sb < num_superblocks; sb++) {
        blocks[sb].d = (__fp16)(prng_rand_uniform(0.0002f, 0.002f));
        blocks[sb].dmin = (__fp16)(prng_rand_uniform(0.0001f, 0.001f));

        uint8_t sc[8], min_val[8];
        for (int j = 0; j < 8; j++) {
            sc[j] = (uint8_t)prng_rand_uniform(1.0f, 63.0f);
            min_val[j] = (uint8_t)prng_rand_uniform(0.0f, 63.0f);
        }

        for (int j = 0; j < 4; j++) {
            blocks[sb].scales[j]     = (sc[j] & 0x3F) | ((min_val[j] >> 4) << 6);
            blocks[sb].scales[j + 4] = (sc[j + 4] & 0x3F) | ((min_val[j + 4] >> 4) << 6);
            blocks[sb].scales[j + 8] = (min_val[j] & 0x0F) | ((min_val[j + 4] & 0x0F) << 4);
        }

        for (int i = 0; i < 128; i++) {
            uint8_t low = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            uint8_t high = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
            blocks[sb].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

inline void generate_ternary_1_58_weights(block_ternary_1_58* blocks, size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = (__fp16)(prng_rand_uniform(0.01f, 0.05f));
        blocks[b]._pad = 0;
        for (int u = 0; u < 2; u++) {
            uint32_t val = 0;
            for (int j = 0; j < 16; j++) {
                int choice = (int)prng_rand_uniform(0.0f, 3.0f);
                if (choice > 2) choice = 2;
                val |= ((uint32_t)choice << (j * 2));
            }
            blocks[b].qs[u] = val;
        }
    }
}

inline void generate_var_rate_affine_weights(block_var_rate_affine* blocks, size_t num_superblocks) {
    const uint32_t sub_offsets[8] = {0, 12, 24, 40, 56, 72, 88, 108};
    const uint8_t sub_bits[8] = {3, 3, 4, 4, 4, 4, 5, 5};

    for (size_t sb = 0; sb < num_superblocks; sb++) {
        blocks[sb].d = (__fp16)prng_rand_uniform(0.0004f, 0.004f);
        blocks[sb].bias = (__fp16)prng_rand_uniform(-0.005f, 0.01f);
        std::memset(blocks[sb]._pad, 0, sizeof(blocks[sb]._pad));

        for (int s = 0; s < 8; s++) {
            blocks[sb].scales[s] = (uint8_t)prng_rand_uniform(10.0f, 210.0f);
            blocks[sb].biases[s] = (uint8_t)prng_rand_uniform(0.0f, 255.0f);
            uint8_t perm = (uint8_t)(s % 3);
            blocks[sb].modes[s] = (perm << 3) | (sub_bits[s] & 0x07);

            uint32_t off = sub_offsets[s];
            uint8_t bits = sub_bits[s];

            if (bits == 3) {
                for (int g = 0; g < 4; g++) {
                    uint8_t q[8];
                    for (int i = 0; i < 8; i++) q[i] = (uint8_t)prng_rand_uniform(0.0f, 8.0f) & 0x07;
                    blocks[sb].qs[off + g * 3 + 0] = q[0] | (q[1] << 3) | ((q[2] & 0x03) << 6);
                    blocks[sb].qs[off + g * 3 + 1] = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 0x01) << 7);
                    blocks[sb].qs[off + g * 3 + 2] = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
                }
            } else if (bits == 5) {
                for (int g = 0; g < 4; g++) {
                    uint8_t q[8];
                    for (int i = 0; i < 8; i++) q[i] = (uint8_t)prng_rand_uniform(0.0f, 32.0f) & 0x1F;
                    blocks[sb].qs[off + g * 5 + 0] = q[0] | ((q[1] & 0x07) << 5);
                    blocks[sb].qs[off + g * 5 + 1] = (q[1] >> 3) | (q[2] << 2) | ((q[3] & 0x01) << 7);
                    blocks[sb].qs[off + g * 5 + 2] = (q[3] >> 1) | ((q[4] & 0x0F) << 4);
                    blocks[sb].qs[off + g * 5 + 3] = (q[4] >> 4) | (q[5] << 1) | ((q[6] & 0x03) << 6);
                    blocks[sb].qs[off + g * 5 + 4] = (q[6] >> 2) | (q[7] << 3);
                }
            } else {
                for (int i = 0; i < 16; i++) {
                    uint8_t low = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
                    uint8_t high = (uint8_t)prng_rand_uniform(0.0f, 16.0f);
                    blocks[sb].qs[off + i] = (high << 4) | (low & 0x0F);
                }
            }
        }
    }
}

inline void generate_exl3_weights(block_exl3* blocks, size_t num_superblocks) {
    for (size_t sb = 0; sb < num_superblocks; sb++) {
        blocks[sb].d = (__fp16)prng_rand_uniform(0.0003f, 0.003f);
        blocks[sb].bias = (__fp16)prng_rand_uniform(-0.004f, 0.008f);
        std::memset(blocks[sb]._pad, 0, sizeof(blocks[sb]._pad));

        for (int i = 0; i < 16; i++) {
            blocks[sb].codebook[i] = (int8_t)prng_rand_uniform(-8.0f, 7.0f);
        }

        for (int s = 0; s < 8; s++) {
            blocks[sb].scales[s] = (uint8_t)prng_rand_uniform(15.0f, 220.0f);
            blocks[sb].residuals[s] = (uint8_t)prng_rand_uniform(0.0f, 255.0f);
            uint32_t off = s * 12;
            for (int g = 0; g < 4; g++) {
                uint8_t q[8];
                for (int i = 0; i < 8; i++) q[i] = (uint8_t)prng_rand_uniform(0.0f, 8.0f) & 0x07;
                blocks[sb].qs[off + g * 3 + 0] = q[0] | (q[1] << 3) | ((q[2] & 0x03) << 6);
                blocks[sb].qs[off + g * 3 + 1] = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 0x01) << 7);
                blocks[sb].qs[off + g * 3 + 2] = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
            }
        }
    }
}

// ============================================================================
// Shader Path Resolver & Pipeline Helper
// ============================================================================
inline std::string find_shader_path(const std::string& filename) {
    if (access(filename.c_str(), R_OK) == 0) return filename;
    std::string parent = "../../" + filename;
    if (access(parent.c_str(), R_OK) == 0) return parent;
    std::string one_up = "../" + filename;
    if (access(one_up.c_str(), R_OK) == 0) return one_up;
    return "/Users/mohammedhossam/Documents/antigravity/wonderful-darwin/" + filename;
}

inline id<MTLLibrary> load_metal_library(id<MTLDevice> device, const std::string& filename) {
    std::string path = find_shader_path(filename);
    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (error) {
        std::cerr << "[ERROR] Could not read shader file " << path << ": "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
    if (error) {
        std::cerr << "[ERROR] Metal shader compilation failed for " << path << ": "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    return lib;
}

inline id<MTLComputePipelineState> create_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, const std::string& func_name) {
    id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:func_name.c_str()]];
    if (!func) {
        std::cerr << "[ERROR] Function not found in library: " << func_name << std::endl;
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    if (error) {
        std::cerr << "[ERROR] Pipeline creation failed for " << func_name << ": "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    return pipeline;
}
