#import "core/bridge/m4_mlx_bridge.h"
#import "core/memory/uma_tracker.h"
#import "core/metal/shader_loader.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#include <filesystem>
#include <dlfcn.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

std::mutex g_bridge_mutex;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_command_queue = nil;
id<MTLLibrary> g_library = nil;
id<MTLComputePipelineState> g_pipelines[6] = {nil, nil, nil, nil, nil, nil};
id<MTLCommandBuffer> g_last_cmd_buffer = nil;
bool g_initialized = false;

static const char* kKernelNames[6] = {
    "quant_router_gemm_q4_0_64x64",
    "quant_router_gemm_mlx_4bit_64x64",
    "quant_router_gemm_q4_k_64x64",
    "quant_router_gemm_ternary_1_58_64x64",
    "quant_router_gemm_var_rate_affine_64x64",
    "quant_router_gemm_exl3_64x64"
};

// Helper to wrap arbitrary host unified memory pointers in MTLBuffer zero-copy
inline id<MTLBuffer> make_zero_copy_buffer(
    id<MTLDevice> device,
    void* ptr,
    size_t bytes,
    NSUInteger& out_offset,
    size_t page_size)
{
    if (!ptr || bytes == 0) return nil;
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t page_base = addr & ~(page_size - 1);
    out_offset = static_cast<NSUInteger>(addr - page_base);
    size_t total_len = out_offset + bytes;

    return [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(page_base)
                                    length:total_len
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
}

// Helper to calculate minimum required weight buffer size per format
size_t get_expected_weight_bytes(int format, uint32_t K, uint32_t N) {
    switch (format) {
        case 0: // QUANT_Q4_0
            return ((size_t)K / 32) * N * 18;
        case 1: // QUANT_MLX_4BIT
            return ((size_t)K / 32) * N * 20;
        case 2: // QUANT_Q4_K
            return ((size_t)K / 256) * N * 144;
        case 3: // QUANT_TERNARY_1_58
            return ((size_t)K / 32) * N * 12;
        case 4: // QUANT_VAR_RATE_AFFINE
            return ((size_t)K / 256) * N * 160;
        case 5: // QUANT_EXL3
            return ((size_t)K / 256) * N * 144;
        default:
            return 0;
    }
}

} // anonymous namespace

size_t m4_bridge_get_expected_weight_bytes(int format, uint32_t K, uint32_t N) {
    return get_expected_weight_bytes(format, K, N);
}

bool m4_bridge_init(const char* metallib_path) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    if (g_initialized) {
        return true;
    }

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "[m4_bridge] Fatal: Failed to acquire system default Metal device." << std::endl;
            return false;
        }

        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "[m4_bridge] Fatal: Failed to create Metal command queue." << std::endl;
            return false;
        }

        NSError* error = nil;
        g_library = nil;

        // 1. Explicit path provided
        if (metallib_path && metallib_path[0] != '\0') {
            std::string path_str(metallib_path);
            if (path_str.rfind(".metallib") != std::string::npos) {
                NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
                g_library = [g_device newLibraryWithURL:url error:&error];
            } else {
                NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metallib_path]
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];
                if (src && !error) {
                    g_library = [g_device newLibraryWithSource:src options:nil error:&error];
                }
            }
        }

        // 2. Default candidate search paths if no path given or explicit failed
        if (!g_library) {
            std::vector<std::string> search_candidates;
            search_candidates.push_back("quant_router_kernels.metal");
            search_candidates.push_back("./quant_router_kernels.metal");

            // Look relative to loaded dynamic library
            Dl_info dlinfo;
            if (dladdr(reinterpret_cast<void*>(m4_bridge_init), &dlinfo) && dlinfo.dli_fname) {
                try {
                    fs::path dylib_dir = fs::path(dlinfo.dli_fname).parent_path();
                    search_candidates.push_back((dylib_dir / "quant_router_kernels.metal").string());
                    search_candidates.push_back((dylib_dir.parent_path() / "quant_router_kernels.metal").string());
                    search_candidates.push_back((dylib_dir.parent_path().parent_path() / "quant_router_kernels.metal").string());
                } catch (...) {}
            }

            for (const auto& cand : search_candidates) {
                if (fs::exists(cand)) {
                    error = nil;
                    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:cand.c_str()]
                                                              encoding:NSUTF8StringEncoding
                                                                 error:&error];
                    if (src && !error) {
                        g_library = [g_device newLibraryWithSource:src options:nil error:&error];
                        if (g_library) {
                            break;
                        }
                    }
                }
            }
        }

        // 3. Fallback to default library bundled in application
        if (!g_library) {
            g_library = [g_device newDefaultLibrary];
        }

        if (!g_library) {
            std::cerr << "[m4_bridge] Fatal: Failed to compile or load Metal shader library." << std::endl;
            if (error) {
                std::cerr << "[m4_bridge] Metal compilation details: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
            return false;
        }

        // Initialize compute pipelines for all 6 quant formats
        for (int i = 0; i < 6; i++) {
            NSString* name = [NSString stringWithUTF8String:kKernelNames[i]];
            id<MTLFunction> fn = [g_library newFunctionWithName:name];
            if (!fn) {
                std::cerr << "[m4_bridge] Fatal: Shader function " << kKernelNames[i]
                          << " not found in library." << std::endl;
                return false;
            }

            error = nil;
            g_pipelines[i] = [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_pipelines[i] || error) {
                std::cerr << "[m4_bridge] Fatal: Failed to build compute pipeline for "
                          << kKernelNames[i] << ": "
                          << (error ? [[error localizedDescription] UTF8String] : "Unknown error")
                          << std::endl;
                return false;
            }
        }

        g_initialized = true;
        return true;
    }
}

int m4_bridge_dispatch_gemm(
    void* X_ptr, size_t X_bytes,
    void* W_ptr, size_t W_bytes,
    void* Y_ptr, size_t Y_bytes,
    int format,
    uint32_t M, uint32_t K, uint32_t N)
{
    if (!g_initialized) {
        return -1;
    }
    if (!X_ptr || !W_ptr || !Y_ptr) {
        return -2;
    }
    // Vulnerability 3: Pointer Alignment Invariant (FP16 elements require at least 2-byte alignment)
    if (((uintptr_t)X_ptr % 2 != 0) || ((uintptr_t)Y_ptr % 2 != 0)) {
        return -2;
    }
    if (format < 0 || format >= 6) {
        return -3;
    }
    if (M == 0 || K == 0 || N == 0) {
        return -4;
    }
    if ((K % 32) != 0) {
        return -4;
    }
    // Vulnerability 2: Super-Block Alignment Invariant (Q4_K, VAR_RATE_AFFINE, EXL3 require K % 256 == 0)
    if ((format == 2 || format == 4 || format == 5) && ((K % 256) != 0)) {
        return -4;
    }
    // Vulnerability 1: Buffer Size Bounds Validation
    size_t min_x_bytes = (size_t)M * K * sizeof(__fp16);
    size_t min_y_bytes = (size_t)M * N * sizeof(__fp16);
    size_t min_w_bytes = get_expected_weight_bytes(format, K, N);
    if (X_bytes < min_x_bytes || Y_bytes < min_y_bytes || W_bytes < min_w_bytes) {
        return -2;
    }

    @autoreleasepool {
        size_t page_size = static_cast<size_t>(getpagesize());
        NSUInteger x_offset = 0;
        NSUInteger w_offset = 0;
        NSUInteger y_offset = 0;

        id<MTLBuffer> x_buf = make_zero_copy_buffer(g_device, X_ptr, X_bytes, x_offset, page_size);
        id<MTLBuffer> w_buf = make_zero_copy_buffer(g_device, W_ptr, W_bytes, w_offset, page_size);
        id<MTLBuffer> y_buf = make_zero_copy_buffer(g_device, Y_ptr, Y_bytes, y_offset, page_size);

        if (!x_buf || !w_buf || !y_buf) {
            std::cerr << "[m4_bridge] Failed to create zero-copy MTLBuffer for inputs/outputs." << std::endl;
            return -5;
        }

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        if (!cmd) {
            return -6;
        }

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) {
            return -6;
        }

        [enc setComputePipelineState:g_pipelines[format]];
        [enc setBuffer:x_buf offset:x_offset atIndex:0];
        [enc setBuffer:w_buf offset:w_offset atIndex:1];
        [enc setBuffer:y_buf offset:y_offset atIndex:2];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [enc setThreadgroupMemoryLength:16384 atIndex:0];

        NSUInteger tg_x = (N + 63) / 64;
        NSUInteger tg_y = (M + 63) / 64;
        [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];

        // Retain zero-copy buffers until GPU execution completes
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
            (void)x_buf;
            (void)w_buf;
            (void)y_buf;
            if (c.status == MTLCommandBufferStatusError) {
                NSLog(@"[m4_bridge] Command buffer failed: %@", c.error);
            }
        }];

        [cmd commit];

        {
            std::lock_guard<std::mutex> lock(g_bridge_mutex);
            g_last_cmd_buffer = cmd;
        }

        return 0;
    }
}

int m4_bridge_dispatch_gemv(
    void* x_ptr, size_t x_bytes,
    void* W_ptr, size_t W_bytes,
    void* y_ptr, size_t y_bytes,
    int format,
    uint32_t K, uint32_t N)
{
    // Autoregressive decode is GEMM with M=1
    return m4_bridge_dispatch_gemm(x_ptr, x_bytes, W_ptr, W_bytes, y_ptr, y_bytes, format, 1, K, N);
}

void m4_bridge_synchronize(void) {
    id<MTLCommandBuffer> cmd_to_wait = nil;
    {
        std::lock_guard<std::mutex> lock(g_bridge_mutex);
        cmd_to_wait = g_last_cmd_buffer;
        g_last_cmd_buffer = nil;
    }

    if (cmd_to_wait != nil) {
        [cmd_to_wait waitUntilCompleted];
        if ([cmd_to_wait status] == MTLCommandBufferStatusError) {
            std::cerr << "[m4_bridge] Cmd error: "
                      << [[[cmd_to_wait error] localizedDescription] UTF8String] << std::endl;
        }
    }
}

double m4_bridge_get_uma_footprint_mb(void) {
    return core::memory::get_uma_phys_footprint_mb();
}

bool m4_bridge_is_initialized(void) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    return g_initialized;
}

void m4_bridge_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    if (g_last_cmd_buffer) {
        [g_last_cmd_buffer waitUntilCompleted];
        g_last_cmd_buffer = nil;
    }
    for (int i = 0; i < 6; i++) {
        g_pipelines[i] = nil;
    }
    g_library = nil;
    g_command_queue = nil;
    g_device = nil;
    g_initialized = false;
}
