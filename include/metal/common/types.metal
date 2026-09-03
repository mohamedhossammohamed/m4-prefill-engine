#pragma once

#include <metal_stdlib>
using namespace metal;

namespace metal_llm {

// Standard unaligned 32-bit integer reader
inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline uint read_u32_unaligned(device const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

} // namespace metal_llm
