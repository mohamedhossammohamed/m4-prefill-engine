// Isolated A/B for the MLP path on M3 Ultra.
//  gate/up+SwiGLU : upstream fused scalar | unfused scalar | simdgroup fused | simdgroup unfused
//  down-projection: upstream scalar       | simdgroup
// The unfused variants exist to settle the port plan's 4c question: is the M4-era
// kernel fusion still a win on a chip where register pressure, not memory passes,
// is the binding constraint?
// ============================================================================
// Part of the M3 Ultra port of the upstream m4-prefill-engine by Mohammed Hossam
// (https://github.com/mohamedhossammohamed/m4-prefill-engine, commit ab01b63,
// Copyright 2026 Mohammed Hossam, licensed under the Apache License 2.0).
//
// This file is new, authored in 2026 by MSW Lab AI, and is licensed under the
// Apache License 2.0 to match the work it accompanies.
//
// Isolated A/B for the MLP path: fused vs unfused, scalar vs simdgroup, plus the
// tile sweep, against a CPU FP32 gold reference.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

struct block_q4_0 { __fp16 d; uint8_t qs[16]; };

static uint32_t prng = 1337;
static inline float ru() { prng = prng * 1664525u + 1013904223u; return (float)prng / (float)0xFFFFFFFF; }
static void gen_act(__fp16* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float u1 = std::max(1e-6f, ru()), u2 = ru();
        p[i] = (__fp16)(std::sqrt(-2.f*std::log(u1))*std::cos(6.2831853f*u2)*0.35f);
    }
}
static void gen_w(block_q4_0* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        b[i].d = (__fp16)(ru()*0.003f + 0.0003f);
        for (int j = 0; j < 16; j++)
            b[i].qs[j] = ((uint8_t)(ru()*16.f) << 4) | ((uint8_t)(ru()*16.f) & 0x0F);
    }
}
static inline float deq(const block_q4_0* W, size_t n, uint K, uint k) {
    const block_q4_0& b = W[n*(K/32) + k/32];
    uint i = k % 32;
    uint nib = (i < 16) ? (b.qs[i] & 0x0F) : (b.qs[i-16] >> 4);
    return ((float)nib - 8.f) * (float)b.d;
}
// CPU FP32 reference. If Wu != null, computes SwiGLU(A@Wg, A@Wu), else plain A@Wg.
static void cpu_ref(const __fp16* A, const block_q4_0* Wg, const block_q4_0* Wu,
                    std::vector<float>& O, uint M, uint N, uint K) {
    O.assign((size_t)M*N, 0.f);
    dispatch_apply(M, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t m) {
        for (uint n = 0; n < N; n++) {
            float g = 0.f, u = 0.f;
            for (uint k = 0; k < K; k++) {
                float a = (float)A[m*K+k];
                g += a * deq(Wg, n, K, k);
                if (Wu) u += a * deq(Wu, n, K, k);
            }
            O[m*N+n] = Wu ? (g/(1.f+std::exp(-g))) * u : g;
        }
    });
}
struct Stat { double med, iqr; };
static Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return { v[n/2], v[(3*n)/4]-v[n/4] };
}

int main(int argc, const char** argv) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* e = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"unified_8b_kernels.metal" encoding:NSUTF8StringEncoding error:&e];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    auto pso = [&](NSString* n) {
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e];
        if (!p) { printf("pso %s: %s\n", [n UTF8String], [[e localizedDescription] UTF8String]); exit(1); }
        return p;
    };
    id<MTLComputePipelineState> P_fused_s = pso(@"fused_gate_up_swiglu_q4_0");
    id<MTLComputePipelineState> P_gemm_s  = pso(@"pipe_gemm_q4_0_32x32");
    id<MTLComputePipelineState> P_fused_g = pso(@"sg_gemm_q4_0_fused_swiglu");
    id<MTLComputePipelineState> P_gemm_g  = pso(@"sg_gemm_q4_0");
    id<MTLComputePipelineState> P_swiglu  = pso(@"swiglu_activation");
    id<MTLComputePipelineState> P_fused64 = pso(@"sg_gemm_q4_0_fused_swiglu_bm64");
    id<MTLComputePipelineState> P_gemm_k64  = pso(@"sg_gemm_q4_0_bk64");
    id<MTLComputePipelineState> P_fused_k64 = pso(@"sg_gemm_q4_0_fused_swiglu_bk64");
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n", [[dev name] UTF8String]);

    auto run = [&](uint M, uint N, uint K, int cfg,
                   id<MTLBuffer> A, id<MTLBuffer> Wg, id<MTLBuffer> Wu,
                   id<MTLBuffer> T0, id<MTLBuffer> T1, id<MTLBuffer> Out) -> double {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
        auto gemm_s = [&](id<MTLBuffer> b, id<MTLBuffer> o) {
            [c setComputePipelineState:P_gemm_s];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:b offset:0 atIndex:1];
            [c setBuffer:o offset:0 atIndex:2];
            [c setBytes:&M length:4 atIndex:3]; [c setBytes:&N length:4 atIndex:4]; [c setBytes:&K length:4 atIndex:5];
            [c setThreadgroupMemoryLength:4096 atIndex:0];
            [c dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+31)/32,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        };
        auto gemm_g = [&](id<MTLBuffer> b, id<MTLBuffer> o) {
            [c setComputePipelineState:P_gemm_g];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:b offset:0 atIndex:1];
            [c setBuffer:o offset:0 atIndex:2];
            [c setBytes:&M length:4 atIndex:3]; [c setBytes:&N length:4 atIndex:4]; [c setBytes:&K length:4 atIndex:5];
            [c setThreadgroupMemoryLength:8192 atIndex:0];   // smA 4K + smB 4K
            [c dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        };
        auto swig = [&]{
            uint ne = M*N;
            [c setComputePipelineState:P_swiglu];
            [c setBuffer:T0 offset:0 atIndex:0]; [c setBuffer:T1 offset:0 atIndex:1];
            [c setBuffer:Out offset:0 atIndex:2]; [c setBytes:&ne length:4 atIndex:3];
            [c dispatchThreads:MTLSizeMake(ne/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        };
        if (cfg == 0) {                       // fused scalar (upstream)
            [c setComputePipelineState:P_fused_s];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:Wg offset:0 atIndex:1];
            [c setBuffer:Wu offset:0 atIndex:2]; [c setBuffer:Out offset:0 atIndex:3];
            [c setBytes:&M length:4 atIndex:4]; [c setBytes:&N length:4 atIndex:5]; [c setBytes:&K length:4 atIndex:6];
            [c setThreadgroupMemoryLength:4096 atIndex:0];
            [c dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+31)/32,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        } else if (cfg == 1) {                // unfused scalar
            gemm_s(Wg, T0); gemm_s(Wu, T1); swig();
        } else if (cfg == 2) {                // fused simdgroup
            [c setComputePipelineState:P_fused_g];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:Wg offset:0 atIndex:1];
            [c setBuffer:Wu offset:0 atIndex:2]; [c setBuffer:Out offset:0 atIndex:3];
            [c setBytes:&M length:4 atIndex:4]; [c setBytes:&N length:4 atIndex:5]; [c setBytes:&K length:4 atIndex:6];
            [c setThreadgroupMemoryLength:10240 atIndex:0];  // smA 2K + smB0 4K + smB1 4K
            [c dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        } else if (cfg == 3) {                // unfused simdgroup
            gemm_g(Wg, T0); gemm_g(Wu, T1); swig();
        } else if (cfg == 7) {                // fused simdgroup, BK=64
            [c setComputePipelineState:P_fused_k64];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:Wg offset:0 atIndex:1];
            [c setBuffer:Wu offset:0 atIndex:2]; [c setBuffer:Out offset:0 atIndex:3];
            [c setBytes:&M length:4 atIndex:4]; [c setBytes:&N length:4 atIndex:5]; [c setBytes:&K length:4 atIndex:6];
            [c setThreadgroupMemoryLength:20480 atIndex:0];
            [c dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        } else if (cfg == 8) {                // plain simdgroup, BK=64
            [c setComputePipelineState:P_gemm_k64];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:Wg offset:0 atIndex:1];
            [c setBuffer:Out offset:0 atIndex:2];
            [c setBytes:&M length:4 atIndex:3]; [c setBytes:&N length:4 atIndex:4]; [c setBytes:&K length:4 atIndex:5];
            [c setThreadgroupMemoryLength:16384 atIndex:0];  // BK=64: smA 8K + smB 8K   // smA 4K + smB 4K
            [c dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        } else if (cfg == 6) {                // fused simdgroup, 64-row tile
            [c setComputePipelineState:P_fused64];
            [c setBuffer:A offset:0 atIndex:0]; [c setBuffer:Wg offset:0 atIndex:1];
            [c setBuffer:Wu offset:0 atIndex:2]; [c setBuffer:Out offset:0 atIndex:3];
            [c setBytes:&M length:4 atIndex:4]; [c setBytes:&N length:4 atIndex:5]; [c setBytes:&K length:4 atIndex:6];
            [c setThreadgroupMemoryLength:32768 atIndex:0];  // BM=64 fused epilogue stages 2*4*64*16 floats
            [c dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        } else if (cfg == 4) {                // plain scalar (down-proj)
            gemm_s(Wg, Out);
        } else {                              // plain simdgroup (down-proj)
            gemm_g(Wg, Out);
        }
        [c endEncoding]; [cb commit]; [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    };

    struct Case { uint M, N, K; bool fused; const char* tag; };
    std::vector<Case> fid;
    for (uint M : {1u,33u,63u,64u,65u,127u,128u,129u,255u,512u}) fid.push_back({M,512,4096,true,"gate/up"});
    for (uint M : {1u,33u,64u,129u,512u}) fid.push_back({M,512,4096,false,"down"});

    printf("\n=== FIDELITY vs CPU FP32 (N=512, K=4096) ===\n");
    printf("%8s %6s | %10s %10s | %10s %10s\n", "case","M","sg maxabs","sg cos","scalar maxabs","scalar cos");
    for (auto& c : fid) {
        size_t na = (size_t)c.M*c.K, nw = (size_t)c.N*(c.K/32), no = (size_t)c.M*c.N;
        id<MTLBuffer> A  = [dev newBufferWithLength:na*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Wg = [dev newBufferWithLength:nw*sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> Wu = [dev newBufferWithLength:nw*sizeof(block_q4_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> T0 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> T1 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> O1 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> O2 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        prng = 1337;
        gen_act((__fp16*)A.contents, na);
        gen_w((block_q4_0*)Wg.contents, nw); gen_w((block_q4_0*)Wu.contents, nw);
        memset(O1.contents,0,no*2); memset(O2.contents,0,no*2);
        run(c.M,c.N,c.K, c.fused?0:4, A,Wg,Wu,T0,T1,O1);
        run(c.M,c.N,c.K, c.fused?2:5, A,Wg,Wu,T0,T1,O2);
        std::vector<float> ref; cpu_ref((const __fp16*)A.contents,(const block_q4_0*)Wg.contents,
                                        c.fused?(const block_q4_0*)Wu.contents:nullptr, ref, c.M,c.N,c.K);
        auto cmp = [&](id<MTLBuffer> b, double& ma, double& cs) {
            const __fp16* o = (const __fp16*)b.contents;
            double m=0,d=0,x=0,y=0;
            for (size_t i=0;i<no;i++){ double a=(double)o[i],r=ref[i]; m=std::max(m,std::abs(a-r)); d+=a*r; x+=a*a; y+=r*r; }
            ma=m; cs=d/(std::sqrt(x)*std::sqrt(y)+1e-30);
        };
        double m1,c1,m2,c2; cmp(O1,m1,c1); cmp(O2,m2,c2);
        printf("%8s %6u | %10.5f %10.6f | %10.5f %10.6f\n", c.tag, c.M, m2, c2, m1, c1);
        fflush(stdout);
    }

    printf("\n=== TIMING at 8B shapes (GPU timestamps, median of 9 after 3 warmup) ===\n");
    const char* names[9] = {"fused-scalar(upstream)","unfused-scalar","fused-sg BM32 BK32","unfused-simdgroup","down-scalar(upstream)","down-sg BM64 BK32","fused-sg BM64 BK32","fused-sg BM32 BK64","down-sg BM64 BK64"};
    for (uint M : {2048u}) {
        for (int part = 0; part < 2; part++) {
            uint N = part ? 4096 : 14336, K = part ? 14336 : 4096;
            printf("\n-- %s  M=%u K=%u N=%u --\n", part?"DOWN PROJECTION":"GATE/UP + SwiGLU", M, K, N);
            size_t na=(size_t)M*K, nw=(size_t)N*(K/32), no=(size_t)M*N;
            id<MTLBuffer> A  = [dev newBufferWithLength:na*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Wg = [dev newBufferWithLength:nw*sizeof(block_q4_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> Wu = [dev newBufferWithLength:nw*sizeof(block_q4_0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> T0 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> T1 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Ob = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Oc = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
            prng = 1337;
            gen_act((__fp16*)A.contents, na);
            gen_w((block_q4_0*)Wg.contents, nw); gen_w((block_q4_0*)Wu.contents, nw);
            int cfgs[2][5] = {{0,2,3,7,-1},{4,5,8,-1,-1}};
            double base = 0;
            for (int ci = 0; ci < 5; ci++) {
                int cfg = cfgs[part][ci];
                if (cfg < 0) continue;
                id<MTLBuffer> out = (ci == 0) ? Ob : Oc;
                memset(out.contents, 0, no*2);
                std::vector<double> t;
                for (int i = 0; i < 12; i++) { double a = run(M,N,K,cfg,A,Wg,Wu,T0,T1,out); if (i>=3) t.push_back(a); }
                Stat s = stats(t);
                if (ci == 0) base = s.med;
                double flops = 2.0*M*N*K*((cfg==4||cfg==5||cfg==8)?1:2);
                double maxd = 0;
                if (ci > 0) {
                    const __fp16* a=(const __fp16*)Ob.contents; const __fp16* b=(const __fp16*)Oc.contents;
                    for (size_t i=0;i<no;i++) maxd = std::max(maxd, (double)std::abs((float)a[i]-(float)b[i]));
                }
                printf("  %-24s %9.3f ms (iqr %6.3f)  %6.2fx  %6.2f TFLOP/s  %s%.5f\n",
                       names[cfg], s.med, s.iqr, base/s.med, flops/(s.med*1e-3)/1e12,
                       ci ? "maxdiff-vs-upstream " : "reference ", maxd);
                fflush(stdout);
            }
        }
    }
    return 0;
} }
