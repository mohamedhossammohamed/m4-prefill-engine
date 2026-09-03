#pragma once

#include <metal_stdlib>
using namespace metal;

namespace metal_llm {
namespace common {

// 32-lane SIMD butterfly reduction for maximum
inline float simd_reduce_max(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return simd_broadcast(val, 0);
}

// 32-lane SIMD butterfly reduction for sum
inline float simd_reduce_sum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return simd_broadcast(val, 0);
}

inline half simd_reduce_sum(half val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return simd_broadcast(val, 0);
}

} // namespace common
} // namespace metal_llm
