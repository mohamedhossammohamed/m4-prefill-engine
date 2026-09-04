#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <dispatch/dispatch.h>
#include "../../quant_router.h"

// ============================================================================
// CPU GOLD REFERENCES (Mathematical Ground Truth)
// ============================================================================

// 1. Q4_0
inline void cpu_gold_reference_q4_0(
    const __fp16* A,
    const block_q4_0* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t nb = K / 32;
    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_q4_0* b_col = B + (size_t)n * nb;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;
            for (uint32_t b = 0; b < nb; b++) {
                double d = (double)b_col[b].d;
                uint32_t a_off = b * 32;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = b_col[b].qs[i];
                    int v0 = (int)(byte_val & 0x0F) - 8;
                    int v1 = (int)(byte_val >> 4) - 8;
                    acc += (double)a_row[a_off + i] * ((double)v0 * d);
                    acc += (double)a_row[a_off + i + 16] * ((double)v1 * d);
                }
            }
            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 2. MLX_4BIT
inline void cpu_gold_reference_mlx_4bit(
    const __fp16* A,
    const block_mlx_4bit* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t nb = K / 32;
    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_mlx_4bit* b_col = B + (size_t)n * nb;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;
            for (uint32_t b = 0; b < nb; b++) {
                double d = (double)b_col[b].d;
                double bias = (double)b_col[b].bias;
                uint32_t a_off = b * 32;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = b_col[b].qs[i];
                    int v0 = (int)(byte_val & 0x0F);
                    int v1 = (int)(byte_val >> 4);
                    double w0 = (double)v0 * d + bias;
                    double w1 = (double)v1 * d + bias;
                    acc += (double)a_row[a_off + i] * w0;
                    acc += (double)a_row[a_off + i + 16] * w1;
                }
            }
            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 3. Q4_K
inline void cpu_gold_reference_q4_k(
    const __fp16* A,
    const block_q4_K* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t n_super = K / 256;
    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_q4_K* b_col = B + (size_t)n * n_super;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;

            for (uint32_t sb = 0; sb < n_super; sb++) {
                const block_q4_K& blk = b_col[sb];
                double d = (double)blk.d;
                double dmin = (double)blk.dmin;

                uint8_t sc[8], min_val[8];
                for (int j = 0; j < 4; j++) {
                    sc[j]     = blk.scales[j] & 0x3F;
                    sc[j + 4] = blk.scales[j + 4] & 0x3F;
                    min_val[j]     = (blk.scales[j + 8] & 0x0F) | ((blk.scales[j] >> 6) << 4);
                    min_val[j + 4] = ((blk.scales[j + 8] >> 4) & 0x0F) | ((blk.scales[j + 4] >> 6) << 4);
                }

                uint32_t sb_k_base = sb * 256;
                for (int s = 0; s < 8; s++) {
                    double d_sub = d * (double)sc[s];
                    double m_sub = dmin * (double)min_val[s];
                    uint32_t sub_k_base = sb_k_base + s * 32;

                    for (int i = 0; i < 16; i++) {
                        uint8_t byte_val = blk.qs[s * 16 + i];
                        int v0 = (int)(byte_val & 0x0F);
                        int v1 = (int)(byte_val >> 4);
                        double w0 = (double)v0 * d_sub - m_sub;
                        double w1 = (double)v1 * d_sub - m_sub;
                        acc += (double)a_row[sub_k_base + i] * w0;
                        acc += (double)a_row[sub_k_base + i + 16] * w1;
                    }
                }
            }
            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 4. TERNARY_1_58
inline void cpu_gold_reference_ternary_1_58(
    const __fp16* A,
    const block_ternary_1_58* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t nb = K / 32;
    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_ternary_1_58* b_col = B + (size_t)n * nb;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;
            for (uint32_t b = 0; b < nb; b++) {
                double d = (double)b_col[b].d;
                uint32_t a_off = b * 32;
                uint32_t q0 = b_col[b].qs[0];
                uint32_t q1 = b_col[b].qs[1];
                for (int j = 0; j < 16; j++) {
                    int c0 = (int)((q0 >> (j * 2)) & 0x3) - 1;
                    acc += (double)a_row[a_off + j] * ((double)c0 * d);
                }
                for (int j = 0; j < 16; j++) {
                    int c1 = (int)((q1 >> (j * 2)) & 0x3) - 1;
                    acc += (double)a_row[a_off + 16 + j] * ((double)c1 * d);
                }
            }
            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 5. VAR_RATE_AFFINE
inline void cpu_gold_reference_var_rate_affine(
    const __fp16* A,
    const block_var_rate_affine* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t n_super = K / 256;
    static const uint32_t sub_offsets[8] = {0, 12, 24, 40, 56, 72, 88, 108};

    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_var_rate_affine* b_col = B + (size_t)n * n_super;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;

            for (uint32_t sb = 0; sb < n_super; sb++) {
                const block_var_rate_affine& blk = b_col[sb];
                double d = (double)blk.d;
                double bias = (double)blk.bias;
                uint32_t sb_k_base = sb * 256;

                for (int s = 0; s < 8; s++) {
                    double sc_raw = (double)blk.scales[s];
                    double bi_raw = (double)blk.biases[s];
                    uint8_t mode = blk.modes[s];

                    double sub_scale = d * sc_raw * 0.0625;
                    double sub_bias = bias + d * (bi_raw - 128.0) * 0.0625;

                    uint8_t bit_depth = mode & 0x07;
                    uint8_t perm_mode = (mode >> 3) & 0x03;

                    uint32_t off = sub_offsets[s];
                    const uint8_t* p = blk.qs + off;
                    double unpacked_w[32];

                    if (bit_depth == 3) {
                        for (int g = 0; g < 4; g++) {
                            uint8_t b0 = p[g * 3 + 0];
                            uint8_t b1 = p[g * 3 + 1];
                            uint8_t b2 = p[g * 3 + 2];

                            uint8_t q0 = (b0) & 0x07;
                            uint8_t q1 = (b0 >> 3) & 0x07;
                            uint8_t q2 = ((b0 >> 6) | (b1 << 2)) & 0x07;
                            uint8_t q3 = (b1 >> 1) & 0x07;
                            uint8_t q4 = (b1 >> 4) & 0x07;
                            uint8_t q5 = ((b1 >> 7) | (b2 << 1)) & 0x07;
                            uint8_t q6 = (b2 >> 2) & 0x07;
                            uint8_t q7 = (b2 >> 5) & 0x07;

                            unpacked_w[g * 8 + 0] = ((double)q0 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 1] = ((double)q1 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 2] = ((double)q2 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 3] = ((double)q3 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 4] = ((double)q4 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 5] = ((double)q5 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 6] = ((double)q6 - 4.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 7] = ((double)q7 - 4.0) * sub_scale + sub_bias;
                        }
                    } else if (bit_depth == 5) {
                        for (int g = 0; g < 4; g++) {
                            uint8_t b0 = p[g * 5 + 0];
                            uint8_t b1 = p[g * 5 + 1];
                            uint8_t b2 = p[g * 5 + 2];
                            uint8_t b3 = p[g * 5 + 3];
                            uint8_t b4 = p[g * 5 + 4];

                            uint8_t q0 = (b0) & 0x1F;
                            uint8_t q1 = ((b0 >> 5) | (b1 << 3)) & 0x1F;
                            uint8_t q2 = (b1 >> 2) & 0x1F;
                            uint8_t q3 = ((b1 >> 7) | (b2 << 1)) & 0x1F;
                            uint8_t q4 = ((b2 >> 4) | (b3 << 4)) & 0x1F;
                            uint8_t q5 = (b3 >> 1) & 0x1F;
                            uint8_t q6 = ((b3 >> 6) | (b4 << 2)) & 0x1F;
                            uint8_t q7 = (b4 >> 3) & 0x1F;

                            unpacked_w[g * 8 + 0] = ((double)q0 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 1] = ((double)q1 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 2] = ((double)q2 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 3] = ((double)q3 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 4] = ((double)q4 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 5] = ((double)q5 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 6] = ((double)q6 - 16.0) * sub_scale + sub_bias;
                            unpacked_w[g * 8 + 7] = ((double)q7 - 16.0) * sub_scale + sub_bias;
                        }
                    } else { // 4-bit
                        for (int i = 0; i < 16; i++) {
                            uint8_t byte_val = p[i];
                            uint8_t q0 = byte_val & 0x0F;
                            uint8_t q1 = byte_val >> 4;
                            unpacked_w[i]      = ((double)q0 - 8.0) * sub_scale + sub_bias;
                            unpacked_w[16 + i] = ((double)q1 - 8.0) * sub_scale + sub_bias;
                        }
                    }

                    uint32_t sub_k_base = sb_k_base + s * 32;
                    for (int j = 0; j < 32; j++) {
                        uint32_t dest_idx = j;
                        if (perm_mode == 1) {
                            dest_idx = ((j & 1) << 4) | (j >> 1);
                        } else if (perm_mode == 2) {
                            dest_idx = 31 - j;
                        }
                        acc += (double)a_row[sub_k_base + dest_idx] * unpacked_w[j];
                    }
                }
            }

            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 6. EXL3
inline void cpu_gold_reference_exl3(
    const __fp16* A,
    const block_exl3* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t n_super = K / 256;

    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_exl3* b_col = B + (size_t)n * n_super;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;

            for (uint32_t sb = 0; sb < n_super; sb++) {
                const block_exl3& blk = b_col[sb];
                double d = (double)blk.d;
                double bias = (double)blk.bias;
                uint32_t sb_k_base = sb * 256;

                for (int s = 0; s < 8; s++) {
                    double sc_raw = (double)blk.scales[s];
                    double res_raw = (double)blk.residuals[s];

                    double sub_scale = d * sc_raw * 0.015625;
                    double sub_res = d * (res_raw - 128.0) * 0.0078125;

                    uint32_t off = s * 12;
                    const uint8_t* p = blk.qs + off;
                    uint32_t sub_k_base = sb_k_base + s * 32;

                    for (int g = 0; g < 4; g++) {
                        uint8_t b0 = p[g * 3 + 0];
                        uint8_t b1 = p[g * 3 + 1];
                        uint8_t b2 = p[g * 3 + 2];

                        uint8_t q0 = (b0) & 0x07;
                        uint8_t q1 = (b0 >> 3) & 0x07;
                        uint8_t q2 = ((b0 >> 6) | (b1 << 2)) & 0x07;
                        uint8_t q3 = (b1 >> 1) & 0x07;
                        uint8_t q4 = (b1 >> 4) & 0x07;
                        uint8_t q5 = ((b1 >> 7) | (b2 << 1)) & 0x07;
                        uint8_t q6 = (b2 >> 2) & 0x07;
                        uint8_t q7 = (b2 >> 5) & 0x07;

                        uint8_t qs[8] = {q0, q1, q2, q3, q4, q5, q6, q7};

                        for (int i = 0; i < 8; i++) {
                            uint8_t q = qs[i];
                            double c_base = (double)blk.codebook[q];
                            double c_res  = (double)blk.codebook[q + 8];
                            double w = bias + sub_scale * c_base + sub_res * c_res;
                            acc += (double)a_row[sub_k_base + g * 8 + i] * w;
                        }
                    }
                }
            }

            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 6b. PRISM_Q2_0 CPU Gold Reference
inline void cpu_gold_reference_prism_q2_0(
    const __fp16* A,
    const block_prism_q2_0* B,
    __fp16* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool direct_head = false,
    uint32_t H = 0,
    uint32_t D = 0)
{
    uint32_t nb = K / 128;
    dispatch_apply(M, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t m) {
        for (uint32_t n = 0; n < N; n++) {
            const block_prism_q2_0* b_col = B + (size_t)n * nb;
            const __fp16* a_row = A + (size_t)m * K;
            double acc = 0.0;
            for (uint32_t b = 0; b < nb; b++) {
                double d = (double)b_col[b].d;
                uint32_t a_off = b * 128;
                for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
                    uint8_t byte_val = b_col[b].qs[byte_idx];
                    for (int j = 0; j < 4; j++) {
                        int code = (int)((byte_val >> (j * 2)) & 0x3) - 1;
                        double w = (double)code * d;
                        acc += (double)a_row[a_off + byte_idx * 4 + j] * w;
                    }
                }
            }
            if (direct_head && H > 0 && D > 0) {
                uint32_t h = n / D;
                uint32_t d_idx = n % D;
                C[(h * M + m) * D + d_idx] = (__fp16)acc;
            } else {
                C[m * N + n] = (__fp16)acc;
            }
        }
    });
}

// 7. Causal Attention CPU Gold Reference
inline void cpu_gold_reference_attention(
    const __fp16* Q, // [H, M, D]
    const __fp16* K, // [H, M, D]
    const __fp16* V, // [H, M, D]
    __fp16* O,       // [H, M, D]
    uint32_t H,
    uint32_t M,
    uint32_t D,
    float scale)
{
    dispatch_apply(H, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t h) {
        const __fp16* q_head = Q + h * M * D;
        const __fp16* k_head = K + h * M * D;
        const __fp16* v_head = V + h * M * D;
        __fp16* o_head = O + h * M * D;

        std::vector<double> scores(M);

        for (uint32_t i = 0; i < M; i++) {
            const __fp16* q_row = q_head + i * D;

            double max_score = -1e30;
            for (uint32_t j = 0; j <= i; j++) {
                const __fp16* k_row = k_head + j * D;
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) {
                    dot += (double)q_row[d] * (double)k_row[d];
                }
                double s = dot * (double)scale;
                scores[j] = s;
                if (s > max_score) max_score = s;
            }

            double sum_exp = 0.0;
            for (uint32_t j = 0; j <= i; j++) {
                double e = std::exp(scores[j] - max_score);
                scores[j] = e;
                sum_exp += e;
            }
            double inv_sum = 1.0 / sum_exp;
            for (uint32_t j = 0; j <= i; j++) {
                scores[j] *= inv_sum;
            }

            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j <= i; j++) {
                    acc += scores[j] * (double)v_head[j * D + d];
                }
                o_head[i * D + d] = (__fp16)acc;
            }
        }
    });
}

// 8. SwiGLU CPU Gold Reference: SiLU(gate) * up
inline void cpu_gold_reference_swiglu(
    const __fp16* gate, // [M, N]
    const __fp16* up,   // [M, N]
    __fp16* out,        // [M, N]
    uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        float g = (float)gate[i];
        float u = (float)up[i];
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = (__fp16)(silu_g * u);
    }
}
