#pragma once

#include <metal_stdlib>
using namespace metal;
#include "../common/types.metal"
#include "../common/simd_reduce.metal"

// ============================================================================
// Gated-Delta-Net Linear Attention Operators
// ============================================================================

// 1. Depthwise Causal 1D Convolution (kernel_size = 4) + SiLU activation
// Input X: [M, C] where C is total conv channels (e.g. 10240)
// Weight: [4, C]
// State:  [3, C] (history for streaming decode)
// Output: [M, C]
kernel void gated_delta_net_conv1d_silu(
    device const half*        X_in       [[buffer(0)]],
    device const half*        weight     [[buffer(1)]],
    device const half*        conv_state [[buffer(2)]],
    device half*              X_out      [[buffer(3)]],
    constant uint&            M          [[buffer(4)]],
    constant uint&            C          [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint t = tid.x; // token position in sequence [0..M-1]
    uint c = tid.y; // channel index [0..C-1]

    if (t >= M || c >= C) return;

    float acc = 0.0f;
    for (int k = 0; k < 4; k++) {
        int src_t = (int)t - (3 - k);
        float in_val = 0.0f;
        if (src_t >= 0) {
            in_val = (float)X_in[src_t * C + c];
        } else if (conv_state != nullptr) {
            int state_idx = 3 + src_t;
            if (state_idx >= 0 && state_idx < 3) {
                in_val = (float)conv_state[state_idx * C + c];
            }
        }
        float w_val = (float)weight[k * C + c];
        acc += in_val * w_val;
    }

    float sig = 1.0f / (1.0f + metal::exp(-acc));
    X_out[t * C + c] = (half)(acc * sig);
}

// 2. Q/K L2-Normalization per head
kernel void gated_delta_net_head_l2norm(
    device half*   X     [[buffer(0)]],
    constant uint& M     [[buffer(1)]],
    constant uint& H     [[buffer(2)]],
    constant uint& D     [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint t = tid.x;
    uint h = tid.y;

    if (t >= M || h >= H) return;

    device half* head_ptr = X + (t * H + h) * D;

    float sum_sq = 0.0f;
    for (uint d = 0; d < D; d++) {
        float v = (float)head_ptr[d];
        sum_sq += v * v;
    }

    float inv_norm = metal::rsqrt(sum_sq + 1e-6f);
    for (uint d = 0; d < D; d++) {
        head_ptr[d] = (half)((float)head_ptr[d] * inv_norm);
    }
}

// 3. Compute Time-Step Decay Gate: g_t = -exp(A) * softplus(a_t + dt_bias)
kernel void gated_delta_net_compute_gate(
    device const half*  a_in    [[buffer(0)]],
    device const half*  A_log   [[buffer(1)]],
    device const half*  dt_bias [[buffer(2)]],
    device half*        g_out   [[buffer(3)]],
    constant uint&      M       [[buffer(4)]],
    constant uint&      H       [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint t = tid.x;
    uint h = tid.y;

    if (t >= M || h >= H) return;

    float a_val = (float)a_in[t * H + h];
    float bias_val = (float)dt_bias[h];
    float a_log_val = (float)A_log[h];

    float x = a_val + bias_val;
    float sp = (x > 20.0f) ? x : metal::log(1.0f + metal::exp(x));
    float g = -metal::exp(a_log_val) * sp;

    g_out[t * H + h] = (half)g;
}

// 4. Recurrent Gated Delta Rule Kernel (sequential in time, parallel across heads)
kernel void gated_delta_net_recurrent_core(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device const half*  g_in      [[buffer(3)]],
    device const half*  b_in      [[buffer(4)]],
    device float*       state_buf [[buffer(5)]], // [H, D, D]
    device half*        O         [[buffer(6)]],
    constant uint&      M         [[buffer(7)]],
    constant uint&      H         [[buffer(8)]],
    constant uint&      D         [[buffer(9)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= H) return;

    float scale = 1.0f / metal::sqrt((float)D);
    device float* S = state_buf + h * D * D;

    for (uint t = 0; t < M; t++) {
        uint tok_head_idx = (t * H + h) * D;
        device const half* q_t = Q + tok_head_idx;
        device const half* k_t = K + tok_head_idx;
        device const half* v_t = V + tok_head_idx;
        device half* o_t = O + tok_head_idx;

        float g_val = (float)g_in[t * H + h];
        float decay = metal::exp(g_val);

        float b_val = (float)b_in[t * H + h];
        float beta_val = 1.0f / (1.0f + metal::exp(-b_val));

        // S_t = S_{t-1} * decay
        for (uint i = 0; i < D; i++) {
            for (uint j = 0; j < D; j++) {
                S[i * D + j] *= decay;
            }
        }

        // kv_mem = S * k_t
        float delta[128];
        for (uint j = 0; j < D; j++) {
            float kv_mem_j = 0.0f;
            for (uint i = 0; i < D; i++) {
                kv_mem_j += S[i * D + j] * (float)k_t[i];
            }
            delta[j] = ((float)v_t[j] - kv_mem_j) * beta_val;
        }

        // S = S + k_t * delta^T
        for (uint i = 0; i < D; i++) {
            float ki = (float)k_t[i];
            for (uint j = 0; j < D; j++) {
                S[i * D + j] += ki * delta[j];
            }
        }

        // o_t = S^T * (q_t * scale)
        for (uint j = 0; j < D; j++) {
            float acc_o = 0.0f;
            for (uint i = 0; i < D; i++) {
                acc_o += S[i * D + j] * ((float)q_t[i] * scale);
            }
            o_t[j] = (half)acc_o;
        }
    }
}

// 5. Gated RMSNorm: RMSNorm(O, eps) * silu(Z)
kernel void gated_delta_net_gated_rmsnorm(
    device half*        O           [[buffer(0)]],
    device const half*  Z           [[buffer(1)]],
    device const half*  norm_weight [[buffer(2)]],
    constant uint&      M           [[buffer(3)]],
    constant uint&      H           [[buffer(4)]],
    constant uint&      D           [[buffer(5)]],
    constant float&     eps         [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint t = tid.x;
    uint h = tid.y;

    if (t >= M || h >= H) return;

    device half* o_ptr = O + (t * H + h) * D;
    device const half* z_ptr = Z + (t * H + h) * D;

    float sum_sq = 0.0f;
    for (uint d = 0; d < D; d++) {
        float v = (float)o_ptr[d];
        sum_sq += v * v;
    }

    float inv_rms = metal::rsqrt(sum_sq / (float)D + eps);

    for (uint d = 0; d < D; d++) {
        float normed = (float)o_ptr[d] * inv_rms * (float)norm_weight[d];
        float z_val = (float)z_ptr[d];
        float silu_z = z_val / (1.0f + metal::exp(-z_val));
        o_ptr[d] = (half)(normed * silu_z);
    }
}
