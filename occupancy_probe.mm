#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
// Occupancy proxy without GPU counters: maxTotalThreadsPerThreadgroup is the compiler's
// register-allocation verdict. If a kernel is register-starved it drops below 1024.
int main() { @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    NSError* e = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"unified_8b_kernels.metal" encoding:NSUTF8StringEncoding error:&e];
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    printf("%-40s %6s %8s %10s\n", "kernel", "maxTG", "simdW", "staticTGmem");
    printf("--------------------------------------------------------------------------\n");
    for (NSString* n in [lib functionNames]) {
        id<MTLFunction> f = [lib newFunctionWithName:n];
        if ([f functionType] != MTLFunctionTypeKernel) continue;
        id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
        if (!p) continue;
        printf("%-40s %6lu %8lu %10lu\n", [n UTF8String],
            (unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.threadExecutionWidth,
            (unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 0;
} }
