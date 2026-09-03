#pragma once

#include <metal_stdlib>
using namespace metal;
#include "types.metal"

namespace metal_llm {
namespace common {

// 128-bit LSU Coalesced Vector Loaders (Apple M4 16-byte burst transaction)
inline void load_128bit_lsu_tile_row(
    device const half* src_row,
    threadgroup half* dst_row,
    uint c_offset,
    uint max_c,
    bool valid_row)
{
    float4 val = float4(0.0f);
    if (valid_row && (c_offset < max_c)) {
        val = *reinterpret_cast<device const float4*>(&src_row[c_offset]);
    }
    *reinterpret_cast<threadgroup float2*>(&dst_row[c_offset])     = val.xy;
    *reinterpret_cast<threadgroup float2*>(&dst_row[c_offset + 4]) = val.zw;
}

} // namespace common
} // namespace metal_llm
