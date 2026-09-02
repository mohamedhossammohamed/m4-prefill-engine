#include <metal_stdlib>
using namespace metal;

// Each thread walks one long contiguous run. Few threads => few loads in flight,
// and adjacent threads touch addresses far apart.
kernel void stream_run(device const float4* src [[buffer(0)]],
                       device float4* sink      [[buffer(1)]],
                       constant uint& n_vec     [[buffer(2)]],
                       constant uint& total_thr [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    uint per = n_vec / total_thr, base = gid * per;
    float4 acc = 0.0f;
    for (uint i = 0; i < per; i++) acc += src[base + i];
    if (acc.x == 12345.6789f) sink[gid] = acc;   // never true; defeats DCE
}

// Same traffic, grid-stride: adjacent threads touch adjacent float4s.
kernel void stream_coalesced(device const float4* src [[buffer(0)]],
                             device float4* sink      [[buffer(1)]],
                             constant uint& n_vec     [[buffer(2)]],
                             constant uint& total_thr [[buffer(3)]],
                             uint gid [[thread_position_in_grid]])
{
    float4 acc = 0.0f;
    for (uint i = gid; i < n_vec; i += total_thr) acc += src[i];
    if (acc.x == 12345.6789f) sink[gid] = acc;
}

// Coalesced read+write, to separate the read-only ceiling from the copy ceiling.
kernel void stream_copy(device const float4* src [[buffer(0)]],
                        device float4* dst       [[buffer(1)]],
                        constant uint& n_vec     [[buffer(2)]],
                        constant uint& total_thr [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
    for (uint i = gid; i < n_vec; i += total_thr) dst[i] = src[i] * 1.0f;
}
