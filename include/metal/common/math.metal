#pragma once

#include <metal_stdlib>
using namespace metal;

namespace metal_llm {
namespace common {

// Numerically robust SiLU: x / (1 + exp(-x))
inline half silu(half x) {
    return x / (1.0h + exp(-x));
}

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline half4 silu4(half4 x) {
    return x / (half4(1.0h) + exp(-x));
}

} // namespace common
} // namespace metal_llm
