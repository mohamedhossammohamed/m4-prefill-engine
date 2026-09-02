#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dispatch/dispatch.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>
#include <functional>

#ifndef BLOCK_Q8_0_DEFINED
#define BLOCK_Q8_0_DEFINED
struct block_q8_0 {
    __fp16 d;
    int8_t qs[32];
};
#endif

class Streaming1MEngine {
public:
    struct RingSlot {
        id<MTLBuffer> bufferK;
        id<MTLBuffer> bufferV;
        id<MTLBuffer> bufferK_q8;
        id<MTLBuffer> bufferV_q8;
        size_t capacityTokens;
        size_t currentTokens;
        size_t chunkStartToken;
        uint32_t chunkIndex;
        bool isQ8_0;
        dispatch_semaphore_t readySem;
        bool isPrefetching;
    };

    struct PerformanceMetrics {
        double latency_ms;         // End-to-end latency (including prefetch I/O)
        double gpu_only_ms;        // GPU compute only latency
        double tflops;             // Achieved TFLOPS
        double ssd_bandwidth_gbps;
        double peak_rss_mb;
        double total_read_bytes;
        double total_read_time_sec;
        double throughput_tok_per_sec;
    };

    Streaming1MEngine(id<MTLDevice> device, id<MTLCommandQueue> queue, uint32_t H, uint32_t D, bool isQ8_0, size_t targetSlotBytes = 128 * 1024 * 1024);
    ~Streaming1MEngine();

    // Initialize pipeline states from MTLLibrary
    bool initializePipelines(id<MTLLibrary> library);

    // Streamlined chunk-by-chunk SSD initialization (Zero host RAM blowup for 1M tokens)
    bool prepareSSDFileChunkedFP16(size_t totalTokens, std::function<void(size_t chunkStart, size_t chunkLen, __fp16* dstK, __fp16* dstV)> generator);
    bool prepareSSDFileChunkedQ8_0(size_t totalTokens, std::function<void(size_t chunkStart, size_t chunkLen, block_q8_0* dstK_q8, block_q8_0* dstV_q8)> generator);

    // Direct RAM-to-SSD preparation (for smaller benchmark sequences)
    bool prepareSSDFileDirectFP16(const __fp16* srcK, const __fp16* srcV, size_t totalTokens);
    bool prepareSSDFileDirectQ8_0(const block_q8_0* srcK_q8, const block_q8_0* srcV_q8, size_t totalTokens);

    // Mode A: Self-Contained Continuous Streaming Attention / Prefill-Decode over 1M SSD KV Cache
    bool executeModeAStreamingAttention(
        id<MTLBuffer> d_Q,
        id<MTLBuffer> d_O,
        size_t totalTokens,
        size_t inRamActiveTokens = 0,
        id<MTLBuffer> d_activeK = nil,
        id<MTLBuffer> d_activeV = nil,
        PerformanceMetrics* metricsOut = nullptr);

    // Mode B: Multi-Token Speculative Parallel Burst Verification over 1M SSD KV Cache
    bool executeModeBSpeculativeVerification(
        id<MTLBuffer> d_Q_spec,
        id<MTLBuffer> d_O_spec,
        uint32_t K_spec,
        size_t pastTokens,
        size_t inRamActiveTokens = 0,
        id<MTLBuffer> d_activeK = nil,
        id<MTLBuffer> d_activeV = nil,
        PerformanceMetrics* metricsOut = nullptr);

    // Asynchronously prefetch chunk from SSD directly into ring buffer slot
    void prefetchChunkAsync(uint32_t slotIdx, uint32_t chunkIdx, size_t chunkStartToken, size_t chunkLen);
    void waitForSlot(uint32_t slotIdx);

    // Accessors
    RingSlot& getSlot(uint32_t slotIdx) { return slots_[slotIdx % 2]; }
    size_t getSlotTokens() const { return slotTokens_; }
    size_t getBytesPerChunk() const { return bytesPerChunk_; }
    double getTotalReadBytes() const { return totalReadBytes_; }
    double getTotalReadTimeSec() const { return totalReadTimeSec_; }
    double getEffectiveReadBandwidthGBps() const {
        return (totalReadTimeSec_ > 0.0) ? ((totalReadBytes_ / 1e9) / totalReadTimeSec_) : 0.0;
    }
    void resetMetrics() {
        totalReadBytes_ = 0.0;
        totalReadTimeSec_ = 0.0;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    uint32_t H_;
    uint32_t D_;
    bool isQ8_0_;
    size_t slotTokens_;

    size_t bytesPerTokenK_;
    size_t bytesPerTokenV_;
    size_t maxChunkBytesK_;
    size_t maxChunkBytesV_;
    size_t bytesPerChunk_;

    int fd_;
    std::string tempFilePath_;

    RingSlot slots_[2];
    dispatch_queue_t ioQueue_;

    double totalReadBytes_;
    double totalReadTimeSec_;

    // Metal Pipeline States
    id<MTLComputePipelineState> psoStreamFP16_;
    id<MTLComputePipelineState> psoStreamQ8_0_;
    id<MTLComputePipelineState> psoSpecVerifyFP16_;
    id<MTLComputePipelineState> psoSpecVerifyQ8_0_;
};
