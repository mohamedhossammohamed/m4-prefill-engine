// Isolated A/B for the causal FlashAttention kernel on M3 Ultra.
// Compares the upstream scalar kernel (flash_attn_fp16_causal_d128) against the
// simdgroup_matrix port (flash_attn_sg_causal_d128): GPU-timestamp medians with
// IQR, plus a CPU FP32 gold reference (max-abs + cosine similarity + NaN scan).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static uint32_t prng = 1337;
static inline float ru() { prng = prng * 1664525u + 1013904223u; return (float)prng / (float)0xFFFFFFFF; }
static void gen(__fp16* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float u1 = std::max(1e-6f, ru()), u2 = ru();
        p[i] = (__fp16)(std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2) * 0.35f);
    }
}

// CPU FP32 causal attention. Q,K,V [H,M,128] -> O [M,H*128]
static void cpu_ref(const __fp16* Q, const __fp16* K, const __fp16* V,
                    std::vector<float>& O, uint M, uint H, float scale) {
    O.assign((size_t)M * H * 128, 0.f);
    dispatch_apply(H, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t h) {
        std::vector<float> s(M);
        for (uint i = 0; i < M; i++) {
            const __fp16* q = Q + ((size_t)h * M + i) * 128;
            float mx = -INFINITY;
            for (uint j = 0; j <= i; j++) {
                const __fp16* k = K + ((size_t)h * M + j) * 128;
                float d = 0.f;
                for (int t = 0; t < 128; t++) d += (float)q[t] * (float)k[t];
                s[j] = d * scale;
                if (s[j] > mx) mx = s[j];
            }
            float sum = 0.f;
            for (uint j = 0; j <= i; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            float inv = sum > 0.f ? 1.f / sum : 0.f;
            float* o = O.data() + (size_t)i * H * 128 + h * 128;
            for (uint j = 0; j <= i; j++) {
                float p = s[j] * inv;
                if (p == 0.f) continue;
                const __fp16* v = V + ((size_t)h * M + j) * 128;
                for (int t = 0; t < 128; t++) o[t] += p * (float)v[t];
            }
        }
    });
}

struct Stat { double med, iqr; };
static Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return { v[n/2], v[(3*n)/4] - v[n/4] };
}

int main(int argc, const char** argv) { @autoreleasepool {
    const uint H = 32, D = 128;
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<uint> Ms = {1, 33, 127, 128, 129, 255, 1023, 1024, 2047, 2048, 4096, 8192};
    uint cpu_max = 2048;
    for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--cpu-max" && i+1 < argc) cpu_max = atoi(argv[++i]);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s   cpu-reference up to M=%u\n\n", [[dev name] UTF8String], cpu_max);
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"unified_8b_kernels.metal" encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("shader compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pOld = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"flash_attn_fp16_causal_d128"] error:&err];
    id<MTLComputePipelineState> pNew = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"flash_attn_sg_causal_d128"] error:&err];
    if (!pOld || !pNew) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    printf("occupancy  old: maxTG=%lu simdWidth=%lu   new: maxTG=%lu simdWidth=%lu\n\n",
        (unsigned long)pOld.maxTotalThreadsPerThreadgroup, (unsigned long)pOld.threadExecutionWidth,
        (unsigned long)pNew.maxTotalThreadsPerThreadgroup, (unsigned long)pNew.threadExecutionWidth);
    id<MTLCommandQueue> q = [dev newCommandQueue];

    printf("%6s | %9s %8s | %9s %8s | %7s | %10s %10s %6s | %s\n",
           "M", "old ms", "iqr", "new ms", "iqr", "speedup", "maxabs", "cos", "nan", "ref");
    printf("-------+--------------------+--------------------+---------+------------------------------+-----\n");

    for (uint M : Ms) {
        size_t nqkv = (size_t)H * M * D, no = (size_t)M * H * D;
        id<MTLBuffer> bQ = [dev newBufferWithLength:nqkv*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bK = [dev newBufferWithLength:nqkv*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV = [dev newBufferWithLength:nqkv*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO1 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO2 = [dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
        prng = 1337;
        gen((__fp16*)bQ.contents, nqkv); gen((__fp16*)bK.contents, nqkv); gen((__fp16*)bV.contents, nqkv);
        memset(bO1.contents, 0, no*2); memset(bO2.contents, 0, no*2);

        auto run = [&](id<MTLComputePipelineState> ps, id<MTLBuffer> out, bool isNew) -> double {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:bQ offset:0 atIndex:0]; [e setBuffer:bK offset:0 atIndex:1];
            [e setBuffer:bV offset:0 atIndex:2]; [e setBuffer:out offset:0 atIndex:3];
            [e setBytes:&M length:4 atIndex:4]; [e setBytes:&H length:4 atIndex:5];
            [e setBytes:&scale length:4 atIndex:6];
            [e setThreadgroupMemoryLength:(isNew ? 20480 : 16384) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((M + 31) / 32, H, 1)
              threadsPerThreadgroup:MTLSizeMake(isNew ? 128 : 32, 1, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        };

        std::vector<double> t1, t2;
        for (int i = 0; i < 12; i++) { double a = run(pOld, bO1, false); if (i >= 3) t1.push_back(a); }
        std::vector<__fp16> snap;
        for (int i = 0; i < 12; i++) {
            double a = run(pNew, bO2, true);  if (i >= 3) t2.push_back(a);
            if (i == 3) { snap.resize(no); memcpy(snap.data(), bO2.contents, no*2); }
        }
        bool det = (memcmp(snap.data(), bO2.contents, no*2) == 0);
        Stat s1 = stats(t1), s2 = stats(t2);

        const __fp16* o1 = (const __fp16*)bO1.contents;
        const __fp16* o2 = (const __fp16*)bO2.contents;
        size_t nan = 0;
        for (size_t i = 0; i < no; i++) if (!std::isfinite((float)o2[i])) nan++;

        double maxabs = 0, dot = 0, n1 = 0, n2 = 0; const char* refname;
        if (M <= cpu_max) {
            std::vector<float> ref; cpu_ref((const __fp16*)bQ.contents, (const __fp16*)bK.contents,
                                            (const __fp16*)bV.contents, ref, M, H, scale);
            for (size_t i = 0; i < no; i++) {
                double a = (double)o2[i], b = ref[i];
                maxabs = std::max(maxabs, std::abs(a - b)); dot += a*b; n1 += a*a; n2 += b*b;
            }
            refname = "cpu-fp32";
        } else {
            for (size_t i = 0; i < no; i++) {
                double a = (double)o2[i], b = (double)o1[i];
                maxabs = std::max(maxabs, std::abs(a - b)); dot += a*b; n1 += a*a; n2 += b*b;
            }
            refname = "vs-old";
        }
        double cosv = dot / (std::sqrt(n1) * std::sqrt(n2) + 1e-30);
        printf("%6u | %9.3f %8.3f | %9.3f %8.3f | %6.2fx | %10.5f %10.6f %6zu | %-8s %s\n",
               M, s1.med, s1.iqr, s2.med, s2.iqr, s1.med / s2.med, maxabs, cosv, nan, refname, det ? "det" : "NONDETERMINISTIC");
        fflush(stdout);
    }
    return 0;
} }
