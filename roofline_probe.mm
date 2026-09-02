// Memory roofline probe. Answers three things the port plan asked for:
//   1. the real bandwidth ceiling, separated into read-only and read+write
//   2. whether there is a cross-die (UltraFusion) scaling knee on an Ultra
//   3. how much the access pattern is worth, independent of thread count
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <vector>
#include <algorithm>

static double run(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                  id<MTLBuffer> a, id<MTLBuffer> b, uint n_vec, uint tg, uint TPG, double mult) {
    uint total = tg * TPG, per = n_vec / total;
    if (!per) return 0;
    double eff = (double)per * total * 16.0 * mult;
    std::vector<double> ts;
    for (int it = 0; it < 7; it++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
        [c setComputePipelineState:p];
        [c setBuffer:a offset:0 atIndex:0]; [c setBuffer:b offset:0 atIndex:1];
        [c setBytes:&n_vec length:4 atIndex:2]; [c setBytes:&total length:4 atIndex:3];
        [c dispatchThreadgroups:MTLSizeMake(tg,1,1) threadsPerThreadgroup:MTLSizeMake(TPG,1,1)];
        [c endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (it >= 2) ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
    }
    std::sort(ts.begin(), ts.end());
    return eff / ts[ts.size()/2] / 1e9;
}

int main(int argc, const char** argv) { @autoreleasepool {
    double theoretical = (argc > 1) ? atof(argv[1]) : 819.2;   // M3 Ultra
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil;
    NSString* s = [NSString stringWithContentsOfFile:@"roofline_probe.metal" encoding:NSUTF8StringEncoding error:&e];
    id<MTLLibrary> lib = [d newLibraryWithSource:s options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLCommandQueue> q = [d newCommandQueue];
    auto pso = [&](NSString* n){ return [d newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e]; };
    id<MTLComputePipelineState> P_run = pso(@"stream_run"), P_co = pso(@"stream_coalesced"), P_cp = pso(@"stream_copy");

    const size_t BYTES = 8ull << 30;
    uint n_vec = (uint)(BYTES / 16);
    id<MTLBuffer> a = [d newBufferWithLength:BYTES options:MTLResourceStorageModePrivate];
    id<MTLBuffer> b = [d newBufferWithLength:BYTES options:MTLResourceStorageModePrivate];
    printf("device: %s   %.1f GB per buffer   theoretical %.1f GB/s\n\n", [[d name] UTF8String], BYTES/1e9, theoretical);

    printf("Scaling sweep -- looking for a cross-die knee (a threadgroup count where scaling degrades)\n");
    printf("%10s %10s %14s %14s\n", "threadgrps", "threads", "coalesced", "run-per-thread");
    printf("---------------------------------------------------------\n");
    for (uint tg : {10u,20u,40u,80u,160u,320u,640u,1280u,2560u,5120u,10240u}) {
        double c1 = run(d,q,P_co,a,b,n_vec,tg,256,1.0), c2 = run(d,q,P_run,a,b,n_vec,tg,256,1.0);
        if (!c1) continue;
        printf("%10u %10u %11.1f GB/s %11.1f GB/s\n", tg, tg*256, c1, c2);
        fflush(stdout);
    }
    printf("\nCeilings (best over thread counts -- each pattern free to use as many threads as it wants)\n");
    double br = 0, bc = 0, bw = 0;
    for (uint tg : {640u, 2560u, 10240u}) {
        br = std::max(br, run(d,q,P_co,a,b,n_vec,tg,256,1.0));
        bw = std::max(bw, run(d,q,P_cp,a,b,n_vec,tg,256,2.0));
        bc = std::max(bc, run(d,q,P_run,a,b,n_vec,tg,256,1.0));
    }
    printf("  read-only,  coalesced : %7.1f GB/s  (%.0f%% of theoretical)\n", br, 100*br/theoretical);
    printf("  read+write, coalesced : %7.1f GB/s  (%.0f%% of theoretical)\n", bw, 100*bw/theoretical);
    printf("  read-only,  scattered : %7.1f GB/s  (%.0f%% of theoretical)\n", bc, 100*bc/theoretical);

    // The comparison that matters for a kernel with a fixed grid: same threads, different
    // pattern. Comparing best-to-best is misleading, because it lets the scattered pattern
    // spend 64x the threads to reach the same place.
    printf("\nAccess pattern at MATCHED thread counts (this is the number a kernel feels)\n");
    printf("%10s %10s %14s %14s %10s\n", "threadgrps", "threads", "coalesced", "scattered", "ratio");
    printf("-------------------------------------------------------------------\n");
    for (uint tg : {80u, 160u, 320u, 1280u, 10240u}) {
        double c1 = run(d,q,P_co,a,b,n_vec,tg,256,1.0), c2 = run(d,q,P_run,a,b,n_vec,tg,256,1.0);
        if (!c1 || !c2) continue;
        printf("%10u %10u %11.1f GB/s %11.1f GB/s %9.2fx\n", tg, tg*256, c1, c2, c1/c2);
    }
    printf("\nThe scattered pattern converges on the coalesced one only once the grid is large\n"
           "enough that each thread's run is short. Below that it costs up to 3x.\n");
    return 0;
} }
