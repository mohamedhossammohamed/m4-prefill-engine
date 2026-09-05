#include <stdint.h>
#include <stddef.h>
#include <dispatch/dispatch.h>
#include <arm_neon.h>

typedef struct {
    __fp16 d;
    uint8_t qs[32];
} block_prism_q2_0;

// Multi-threaded dequantize N blocks of PRISM_Q2_0 into __fp16 array
void dequantize_prism_q2_0_fp16(const void* raw_blocks, __fp16* out_fp16, size_t n_blocks) {
    const block_prism_q2_0* blocks = (const block_prism_q2_0*)raw_blocks;
    size_t chunk_size = 2048;
    size_t n_chunks = (n_blocks + chunk_size - 1) / chunk_size;

    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk_idx) {
        size_t start_b = chunk_idx * chunk_size;
        size_t end_b = start_b + chunk_size;
        if (end_b > n_blocks) end_b = n_blocks;

        for (size_t b = start_b; b < end_b; b++) {
            __fp16 d = blocks[b].d;
            __fp16* out = out_fp16 + (b * 128);
            const uint8_t* qs = blocks[b].qs;

            #pragma clang loop unroll(full)
            for (int i = 0; i < 32; i++) {
                uint8_t byte = qs[i];
                out[i * 4 + 0] = ((int)(byte & 0x03) - 1) * d;
                out[i * 4 + 1] = ((int)((byte >> 2) & 0x03) - 1) * d;
                out[i * 4 + 2] = ((int)((byte >> 4) & 0x03) - 1) * d;
                out[i * 4 + 3] = ((int)((byte >> 6) & 0x03) - 1) * d;
            }
        }
    });
}

// High-performance ARM NEON GEMV for PRISM_Q2_0 weights (Native FP16 SIMD)
void gemv_prism_q2_0(const __fp16* X, const void* B_raw, __fp16* Y, uint32_t K, uint32_t N) {
    const block_prism_q2_0* B = (const block_prism_q2_0*)B_raw;
    uint32_t n_blocks_k = K / 128;
    size_t chunk_cols = 128;
    size_t n_chunks = (N + chunk_cols - 1) / chunk_cols;

    static const int8_t tbl_data[16] = {-1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int8x16_t lut_tbl = vld1q_s8(tbl_data);
    uint8x16_t mask_3 = vdupq_n_u8(3);

    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk_idx) {
        size_t start_col = chunk_idx * chunk_cols;
        size_t end_col = start_col + chunk_cols;
        if (end_col > N) end_col = N;

        for (size_t col = start_col; col < end_col; col++) {
            const block_prism_q2_0* col_blocks = B + col * n_blocks_k;
            float total_dot = 0.0f;

            for (uint32_t b = 0; b < n_blocks_k; b++) {
                float d = (float)col_blocks[b].d;
                const __fp16* x_b = X + b * 128;
                const uint8_t* qs = col_blocks[b].qs;

                float16x8_t acc0 = vdupq_n_f16(0.0);
                float16x8_t acc1 = vdupq_n_f16(0.0);
                float16x8_t acc2 = vdupq_n_f16(0.0);
                float16x8_t acc3 = vdupq_n_f16(0.0);

                for (int h = 0; h < 2; h++) {
                    uint8x16_t raw_bytes = vld1q_u8(qs + h * 16);
                    const __fp16* x_ptr = x_b + h * 64;

                    // Direct hardware 4-way de-interleave of X into register quad
                    float16x8x4_t x0_4 = vld4q_f16(x_ptr);
                    float16x8x4_t x1_4 = vld4q_f16(x_ptr + 32);

                    uint8x16_t c0 = vandq_u8(raw_bytes, mask_3);
                    uint8x16_t c1 = vandq_u8(vshrq_n_u8(raw_bytes, 2), mask_3);
                    uint8x16_t c2 = vandq_u8(vshrq_n_u8(raw_bytes, 4), mask_3);
                    uint8x16_t c3 = vshrq_n_u8(raw_bytes, 6);

                    int8x16_t w0 = vqtbl1q_s8(lut_tbl, c0);
                    int8x16_t w1 = vqtbl1q_s8(lut_tbl, c1);
                    int8x16_t w2 = vqtbl1q_s8(lut_tbl, c2);
                    int8x16_t w3 = vqtbl1q_s8(lut_tbl, c3);

                    // Dual-issue Native FP16 FMAs
                    float16x8_t wf0_low = vcvtq_f16_s16(vmovl_s8(vget_low_s8(w0)));
                    float16x8_t wf0_high = vcvtq_f16_s16(vmovl_s8(vget_high_s8(w0)));
                    acc0 = vfmaq_f16(acc0, x0_4.val[0], wf0_low);
                    acc0 = vfmaq_f16(acc0, x1_4.val[0], wf0_high);

                    float16x8_t wf1_low = vcvtq_f16_s16(vmovl_s8(vget_low_s8(w1)));
                    float16x8_t wf1_high = vcvtq_f16_s16(vmovl_s8(vget_high_s8(w1)));
                    acc1 = vfmaq_f16(acc1, x0_4.val[1], wf1_low);
                    acc1 = vfmaq_f16(acc1, x1_4.val[1], wf1_high);

                    float16x8_t wf2_low = vcvtq_f16_s16(vmovl_s8(vget_low_s8(w2)));
                    float16x8_t wf2_high = vcvtq_f16_s16(vmovl_s8(vget_high_s8(w2)));
                    acc2 = vfmaq_f16(acc2, x0_4.val[2], wf2_low);
                    acc2 = vfmaq_f16(acc2, x1_4.val[2], wf2_high);

                    float16x8_t wf3_low = vcvtq_f16_s16(vmovl_s8(vget_low_s8(w3)));
                    float16x8_t wf3_high = vcvtq_f16_s16(vmovl_s8(vget_high_s8(w3)));
                    acc3 = vfmaq_f16(acc3, x0_4.val[3], wf3_low);
                    acc3 = vfmaq_f16(acc3, x1_4.val[3], wf3_high);
                }

                float16x8_t sum_vec = vaddq_f16(vaddq_f16(acc0, acc1), vaddq_f16(acc2, acc3));
                float16x4_t s4 = vadd_f16(vget_low_f16(sum_vec), vget_high_f16(sum_vec));
                s4 = vpadd_f16(s4, s4);
                s4 = vpadd_f16(s4, s4);
                float block_sum = (float)vget_lane_f16(s4, 0);
                total_dot += block_sum * d;
            }
            Y[col] = (__fp16)total_dot;
        }
    });
}

// Batch GEMM on PRISM_Q2_0 weights for Prefill (M >= 1)
void gemm_prism_q2_0(const __fp16* X, const void* B_raw, __fp16* Y, uint32_t M, uint32_t K, uint32_t N) {
    if (M == 1) {
        gemv_prism_q2_0(X, B_raw, Y, K, N);
        return;
    }
    for (uint32_t m = 0; m < M; m++) {
        gemv_prism_q2_0(X + (size_t)m * K, B_raw, Y + (size_t)m * N, K, N);
    }
}
