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

#ifndef BLOCK_Q8_0_DEFINED
#define BLOCK_Q8_0_DEFINED
struct block_q8_0 {
    __fp16 d;
    int8_t qs[32];
};
#endif

class AsyncKVRingBuffer {
public:
    struct Slot {
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

    AsyncKVRingBuffer(id<MTLDevice> device, uint32_t H, uint32_t D, size_t slotTokens, bool isQ8_0);
    ~AsyncKVRingBuffer();

    // Prepare unlinked NVMe SSD file with KV cache data (Strict zero disk litter)
    bool prepareSSDFileFP16(const __fp16* srcK, const __fp16* srcV, size_t totalTokens);
    bool prepareSSDFileQ8_0(const block_q8_0* srcK_q8, const block_q8_0* srcV_q8, size_t totalTokens);

    // Asynchronously prefetch chunk from SSD directly into ring buffer slot
    void prefetchChunkAsync(uint32_t slotIdx, uint32_t chunkIdx, size_t chunkStartToken, size_t chunkLen);

    // Synchronize slot I/O before GPU consumption
    void waitForSlot(uint32_t slotIdx);

    // Accessors
    Slot& getSlot(uint32_t slotIdx) { return slots_[slotIdx % 2]; }
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
    uint32_t H_;
    uint32_t D_;
    size_t slotTokens_;
    bool isQ8_0_;

    size_t bytesPerTokenK_;
    size_t bytesPerTokenV_;
    size_t maxChunkBytesK_;
    size_t maxChunkBytesV_;
    size_t bytesPerChunk_;

    int fd_;
    std::string tempFilePath_;

    Slot slots_[2];
    dispatch_queue_t ioQueue_;

    double totalReadBytes_;
    double totalReadTimeSec_;
};
