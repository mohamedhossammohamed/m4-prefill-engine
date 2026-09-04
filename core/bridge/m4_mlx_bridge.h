#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ZERO-COPY METAL UMA BRIDGE C-ABI INTERFACE
// ============================================================================

/**
 * Initializes the Metal device, command queue, and compiles/loads the
 * Universal Quantization Router shaders.
 * 
 * @param metallib_path Optional file path to a compiled .metallib or source .metal file.
 *                      If NULL or empty string, searches default locations for quant_router_kernels.metal.
 * @return true if initialization succeeded, false otherwise.
 */
bool m4_bridge_init(const char* metallib_path);

/**
 * Calculates the expected byte size of the quantized weight buffer W.
 *
 * @param format Quantization format enum (0..5)
 * @param K      Input channel dimension
 * @param N      Output channel dimension
 * @return Expected minimum byte size, or 0 if format/dimensions invalid.
 */
size_t m4_bridge_get_expected_weight_bytes(int format, uint32_t K, uint32_t N);

/**
 * Wraps host unified memory pointers zero-copy in MTLResourceStorageModeShared buffers
 * and dispatches the corresponding router GEMM kernel for prefill.
 *
 * @param X_ptr   Pointer to input activation matrix X [M, K] (__fp16)
 * @param X_bytes Byte size of buffer X (at least M * K * sizeof(__fp16))
 * @param W_ptr   Pointer to quantized weight matrix W [K, N]
 * @param W_bytes Byte size of buffer W
 * @param Y_ptr   Pointer to output matrix Y [M, N] (__fp16)
 * @param Y_bytes Byte size of buffer Y (at least M * N * sizeof(__fp16))
 * @param format  Quantization format enum (0: Q4_0, 1: MLX_4BIT, 2: Q4_K, 3: TERNARY_1_58, 4: VAR_RATE_AFFINE, 5: EXL3)
 * @param M       Sequence length / token rows
 * @param K       Input channel dimension
 * @param N       Output channel dimension
 * @return 0 on success, negative error code on failure:
 *         -1: Bridge not initialized
 *         -2: Null pointer, unaligned pointer, or insufficient buffer size
 *         -3: Invalid quantization format
 *         -4: Invalid dimension (0, non-multiple of 32, or non-multiple of 256 for super-blocks)
 *         -5: Metal zero-copy buffer allocation failure
 *         -6: Metal command buffer or compute encoder failure
 */
int m4_bridge_dispatch_gemm(
    void* X_ptr, size_t X_bytes,
    void* W_ptr, size_t W_bytes,
    void* Y_ptr, size_t Y_bytes,
    int format,
    uint32_t M, uint32_t K, uint32_t N
);

/**
 * Wraps host unified memory pointers zero-copy and dispatches decode GEMV kernel.
 *
 * @param x_ptr   Pointer to input activation vector x [1, K] (__fp16)
 * @param x_bytes Byte size of vector x (at least K * sizeof(__fp16))
 * @param W_ptr   Pointer to quantized weight matrix W [K, N]
 * @param W_bytes Byte size of buffer W
 * @param y_ptr   Pointer to output vector y [1, N] (__fp16)
 * @param y_bytes Byte size of vector y (at least N * sizeof(__fp16))
 * @param format  Quantization format enum
 * @param K       Input channel dimension
 * @param N       Output channel dimension
 * @return 0 on success, non-zero error code on failure.
 */
int m4_bridge_dispatch_gemv(
    void* x_ptr, size_t x_bytes,
    void* W_ptr, size_t W_bytes,
    void* y_ptr, size_t y_bytes,
    int format,
    uint32_t K, uint32_t N
);

/**
 * Synchronizes GPU command queue, blocking until all in-flight Metal command buffers finish.
 */
void m4_bridge_synchronize(void);

/**
 * Returns the exact physical unified memory footprint of the current process in megabytes (MB)
 * via task_vm_info.phys_footprint.
 */
double m4_bridge_get_uma_footprint_mb(void);

/**
 * Checks if the Metal bridge has been initialized.
 */
bool m4_bridge_is_initialized(void);

/**
 * Releases cached Metal resources and resets bridge state.
 */
void m4_bridge_shutdown(void);

#ifdef __cplusplus
}
#endif
