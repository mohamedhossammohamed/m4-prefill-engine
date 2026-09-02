// ============================================================================
// Part of the M3 Ultra port of the upstream m4-prefill-engine by Mohammed Hossam
// (https://github.com/mohamedhossammohamed/m4-prefill-engine, commit ab01b63,
// Copyright 2026 Mohammed Hossam, licensed under the Apache License 2.0).
//
// This file is new, authored in 2026 by MSW Lab AI, and is licensed under the
// Apache License 2.0 to match the work it accompanies.
//
// Compile-time capability probe for MSL simdgroup_matrix on the target device.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
static void tryc(id<MTLDevice> dev, const char* name, const char* src) {
    NSError* e = nil;
    id<MTLLibrary> l = [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&e];
    printf("%-34s %s\n", name, l ? "OK" : [[e localizedDescription] UTF8String]);
}
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", [[dev name] UTF8String]);
    printf("maxThreadgroupMemory: %lu B\n", (unsigned long)[dev maxThreadgroupMemoryLength]);
    printf("maxThreadsPerThreadgroup: %lu\n", (unsigned long)[dev maxThreadsPerThreadgroup].width);
    printf("supportsFamily(Apple9): %d\n", (int)[dev supportsFamily:MTLGPUFamilyApple9]);

    const char* hdr = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n";

    tryc(dev, "half*half -> half acc", [NSString stringWithFormat:@"%s%s", hdr,
      "kernel void k(device half* o, threadgroup half* s [[threadgroup(0)]]) {\n"
      "  simdgroup_half8x8 a,b; simdgroup_half8x8 c = make_filled_simdgroup_matrix<half,8,8>(0.h);\n"
      "  simdgroup_load(a,s,8); simdgroup_load(b,s,8,ulong2(0,0),true);\n"
      "  simdgroup_multiply_accumulate(c,a,b,c); simdgroup_store(c,o,8); }"].UTF8String);

    tryc(dev, "half*half -> FLOAT acc", [NSString stringWithFormat:@"%s%s", hdr,
      "kernel void k(device float* o, threadgroup half* s [[threadgroup(0)]]) {\n"
      "  simdgroup_half8x8 a,b; simdgroup_float8x8 c = make_filled_simdgroup_matrix<float,8,8>(0.f);\n"
      "  simdgroup_load(a,s,8); simdgroup_load(b,s,8,ulong2(0,0),true);\n"
      "  simdgroup_multiply_accumulate(c,a,b,c); simdgroup_store(c,o,8); }"].UTF8String);

    tryc(dev, "float*float -> float acc", [NSString stringWithFormat:@"%s%s", hdr,
      "kernel void k(device float* o, threadgroup float* s [[threadgroup(0)]]) {\n"
      "  simdgroup_float8x8 a,b; simdgroup_float8x8 c = make_filled_simdgroup_matrix<float,8,8>(0.f);\n"
      "  simdgroup_load(a,s,8); simdgroup_load(b,s,8);\n"
      "  simdgroup_multiply(c,a,b); simdgroup_store(c,o,8); }"].UTF8String);

    tryc(dev, "simdgroup_index attr + simd_shuffle_xor", [NSString stringWithFormat:@"%s%s", hdr,
      "kernel void k(device float* o, uint sg [[simdgroup_index_in_threadgroup]],\n"
      "  uint ns [[simdgroups_per_threadgroup]], uint l [[thread_index_in_simdgroup]]) {\n"
      "  float v = o[l]; v = max(v, simd_shuffle_xor(v,1u)); v += simd_shuffle_xor(v,2u); o[sg*ns+l]=v; }"].UTF8String);

    tryc(dev, "sg_store to threadgroup float", [NSString stringWithFormat:@"%s%s", hdr,
      "kernel void k(threadgroup float* s [[threadgroup(0)]], threadgroup half* p [[threadgroup(1)]]) {\n"
      "  simdgroup_float8x8 c = make_filled_simdgroup_matrix<float,8,8>(1.f);\n"
      "  simdgroup_store(c, s, 16); simdgroup_half8x8 h; simdgroup_load(h, p, 16); }"].UTF8String);
    return 0;
} }
