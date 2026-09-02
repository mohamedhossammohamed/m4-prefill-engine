#import "streaming_1m_engine.h"
#include <mach/mach.h>
#include <mach/task_info.h>

static double get_current_rss_mb() {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count);
    if (kr != KERN_SUCCESS) return 0.0;
    return (double)vm_info.phys_footprint / (1024.0 * 1024.0);
}

Streaming1MEngine::Streaming1MEngine(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    uint32_t H,
    uint32_t D,
    bool isQ8_0,
    size_t targetSlotBytes)
    : device_(device),
      queue_(queue),
      H_(H),
      D_(D),
      isQ8_0_(isQ8_0),
      fd_(-1),
      totalReadBytes_(0.0),
      totalReadTimeSec_(0.0),
      psoStreamFP16_(nil),
      psoStreamQ8_0_(nil),
      psoSpecVerifyFP16_(nil),
      psoSpecVerifyQ8_0_(nil)
{
    ioQueue_ = dispatch_queue_create("com.m4engine.streaming_1m_nvme_io", DISPATCH_QUEUE_SERIAL);

    if (isQ8_0_) {
        size_t blocksPerToken = D_ / 32;
        bytesPerTokenK_ = H_ * blocksPerToken * sizeof(block_q8_0);
        bytesPerTokenV_ = H_ * blocksPerToken * sizeof(block_q8_0);
    } else {
        bytesPerTokenK_ = H_ * D_ * sizeof(__fp16);
        bytesPerTokenV_ = H_ * D_ * sizeof(__fp16);
    }

    // Determine slot tokens to fit targetSlotBytes (e.g. 128MB per slot)
    size_t totalBytesPerToken = bytesPerTokenK_ + bytesPerTokenV_;
    size_t rawTokens = targetSlotBytes / totalBytesPerToken;
    // Align to multiple of 64 tokens (FlashAttention tile size)
    slotTokens_ = (rawTokens / 64) * 64;
    if (slotTokens_ == 0) slotTokens_ = 64;

    maxChunkBytesK_ = slotTokens_ * bytesPerTokenK_;
    maxChunkBytesV_ = slotTokens_ * bytesPerTokenV_;
    bytesPerChunk_  = maxChunkBytesK_ + maxChunkBytesV_;

    // Allocate Dual Unified Memory Ring Buffer Slots (Shared Memory Mode for zero-copy GPU access)
    for (int i = 0; i < 2; i++) {
        slots_[i].capacityTokens = slotTokens_;
        slots_[i].currentTokens = 0;
        slots_[i].chunkStartToken = 0;
        slots_[i].chunkIndex = 0;
        slots_[i].isQ8_0 = isQ8_0_;
        slots_[i].readySem = dispatch_semaphore_create(0);
        slots_[i].isPrefetching = false;

        if (isQ8_0_) {
            slots_[i].bufferK_q8 = [device_ newBufferWithLength:maxChunkBytesK_ options:MTLResourceStorageModeShared];
            slots_[i].bufferV_q8 = [device_ newBufferWithLength:maxChunkBytesV_ options:MTLResourceStorageModeShared];
            slots_[i].bufferK = nil;
            slots_[i].bufferV = nil;
        } else {
            slots_[i].bufferK = [device_ newBufferWithLength:maxChunkBytesK_ options:MTLResourceStorageModeShared];
            slots_[i].bufferV = [device_ newBufferWithLength:maxChunkBytesV_ options:MTLResourceStorageModeShared];
            slots_[i].bufferK_q8 = nil;
            slots_[i].bufferV_q8 = nil;
        }
    }
}

Streaming1MEngine::~Streaming1MEngine() {
    if (fd_ >= 0) {
        ftruncate(fd_, 0); // Release APFS block extents immediately
        close(fd_);
        fd_ = -1;
    }
    for (int i = 0; i < 2; i++) {
        slots_[i].bufferK = nil;
        slots_[i].bufferV = nil;
        slots_[i].bufferK_q8 = nil;
        slots_[i].bufferV_q8 = nil;
    }
}

bool Streaming1MEngine::initializePipelines(id<MTLLibrary> library) {
    NSError* error = nil;
    auto createPSO = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (!fn) {
            std::cerr << "[-] Error: Kernel function " << [name UTF8String] << " not found in library." << std::endl;
            return nil;
        }
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&error];
        if (error || !pso) {
            std::cerr << "[-] Error compiling PSO for " << [name UTF8String] << ": " << [[error localizedDescription] UTF8String] << std::endl;
            return nil;
        }
        return pso;
    };

    if (D_ == 64) {
        psoStreamFP16_     = createPSO(@"streaming_1m_flash_attn_chunk_fp16_d64");
        psoStreamQ8_0_     = createPSO(@"streaming_1m_flash_attn_chunk_q8_0_d64");
        psoSpecVerifyFP16_ = createPSO(@"streaming_1m_speculative_verify_fp16_d64");
        psoSpecVerifyQ8_0_ = createPSO(@"streaming_1m_speculative_verify_q8_0_d64");
    } else if (D_ == 128) {
        psoStreamFP16_     = createPSO(@"streaming_1m_flash_attn_chunk_fp16_d128");
        psoStreamQ8_0_     = createPSO(@"streaming_1m_flash_attn_chunk_q8_0_d128");
        psoSpecVerifyFP16_ = createPSO(@"streaming_1m_speculative_verify_fp16_d128");
        psoSpecVerifyQ8_0_ = createPSO(@"streaming_1m_speculative_verify_q8_0_d128");
    } else {
        std::cerr << "[-] Unsupported head dimension D=" << D_ << " (Supported: 64, 128)" << std::endl;
        return false;
    }

    return (psoStreamFP16_ && psoStreamQ8_0_ && psoSpecVerifyFP16_ && psoSpecVerifyQ8_0_);
}

bool Streaming1MEngine::prepareSSDFileChunkedFP16(
    size_t totalTokens,
    std::function<void(size_t chunkStart, size_t chunkLen, __fp16* dstK, __fp16* dstV)> generator)
{
    if (fd_ >= 0) {
        ftruncate(fd_, 0);
        close(fd_);
        fd_ = -1;
    }

    char tempPath[PATH_MAX];
    snprintf(tempPath, sizeof(tempPath), "/tmp/streaming_1m_fp16_%d_%p.bin", getpid(), this);
    tempFilePath_ = tempPath;

    fd_ = open(tempFilePath_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) {
        perror("[-] Failed to open temporary NVMe SSD file for KV cache");
        return false;
    }

    // STRICT INVARIANT 4: Unlink immediately after open so macOS guarantees zero disk litter
    unlink(tempFilePath_.c_str());

    // Disable macOS Unified Buffer Cache to force direct hardware NVMe I/O
    if (fcntl(fd_, F_NOCACHE, 1) < 0) {
        perror("[-] Warning: Failed to set F_NOCACHE direct I/O on SSD file");
    }

    size_t numChunks = (totalTokens + slotTokens_ - 1) / slotTokens_;
    size_t chunk_bytes_k = H_ * slotTokens_ * D_ * sizeof(__fp16);
    size_t chunk_bytes_v = H_ * slotTokens_ * D_ * sizeof(__fp16);
    void* stagingK_raw = nullptr;
    void* stagingV_raw = nullptr;
    posix_memalign(&stagingK_raw, 16384, chunk_bytes_k);
    posix_memalign(&stagingV_raw, 16384, chunk_bytes_v);
    auto stagingK = static_cast<__fp16*>(stagingK_raw);
    auto stagingV = static_cast<__fp16*>(stagingV_raw);

    for (size_t c = 0; c < numChunks; c++) {
        size_t chunkStart = c * slotTokens_;
        size_t chunkLen = std::min(slotTokens_, totalTokens - chunkStart);

        // Fill staging buffer using chunk generator
        generator(chunkStart, chunkLen, stagingK, stagingV);

        off_t offK = (off_t)c * bytesPerChunk_;
        off_t offV = (off_t)c * bytesPerChunk_ + (off_t)maxChunkBytesK_;

        size_t writeBytesK = H_ * slotTokens_ * D_ * sizeof(__fp16);
        size_t writeBytesV = H_ * slotTokens_ * D_ * sizeof(__fp16);

        ssize_t wk = pwrite(fd_, stagingK, writeBytesK, offK);
        ssize_t wv = pwrite(fd_, stagingV, writeBytesV, offV);

        if (wk != (ssize_t)writeBytesK || wv != (ssize_t)writeBytesV) {
            perror("[-] Failed to write chunk to NVMe SSD file");
            free(stagingK_raw);
            free(stagingV_raw);
            return false;
        }
    }

    free(stagingK_raw);
    free(stagingV_raw);
    fsync(fd_);
    return true;
}

bool Streaming1MEngine::prepareSSDFileChunkedQ8_0(
    size_t totalTokens,
    std::function<void(size_t chunkStart, size_t chunkLen, block_q8_0* dstK_q8, block_q8_0* dstV_q8)> generator)
{
    if (fd_ >= 0) {
        ftruncate(fd_, 0);
        close(fd_);
        fd_ = -1;
    }

    char tempPath[PATH_MAX];
    snprintf(tempPath, sizeof(tempPath), "/tmp/streaming_1m_q8_0_%d_%p.bin", getpid(), this);
    tempFilePath_ = tempPath;

    fd_ = open(tempFilePath_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) {
        perror("[-] Failed to open temporary NVMe SSD file for Q8_0 KV cache");
        return false;
    }

    unlink(tempFilePath_.c_str());

    if (fcntl(fd_, F_NOCACHE, 1) < 0) {
        perror("[-] Warning: Failed to set F_NOCACHE direct I/O on SSD file");
    }

    size_t blocksPerToken = D_ / 32;
    size_t numChunks = (totalTokens + slotTokens_ - 1) / slotTokens_;
    size_t chunk_bytes_k = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);
    size_t chunk_bytes_v = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);
    void* stagingK_raw = nullptr;
    void* stagingV_raw = nullptr;
    posix_memalign(&stagingK_raw, 16384, chunk_bytes_k);
    posix_memalign(&stagingV_raw, 16384, chunk_bytes_v);
    auto stagingK = static_cast<block_q8_0*>(stagingK_raw);
    auto stagingV = static_cast<block_q8_0*>(stagingV_raw);

    for (size_t c = 0; c < numChunks; c++) {
        size_t chunkStart = c * slotTokens_;
        size_t chunkLen = std::min(slotTokens_, totalTokens - chunkStart);

        generator(chunkStart, chunkLen, stagingK, stagingV);

        off_t offK = (off_t)c * bytesPerChunk_;
        off_t offV = (off_t)c * bytesPerChunk_ + (off_t)maxChunkBytesK_;

        size_t writeBytesK = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);
        size_t writeBytesV = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);

        ssize_t wk = pwrite(fd_, stagingK, writeBytesK, offK);
        ssize_t wv = pwrite(fd_, stagingV, writeBytesV, offV);

        if (wk != (ssize_t)writeBytesK || wv != (ssize_t)writeBytesV) {
            perror("[-] Failed to write Q8_0 chunk to NVMe SSD file");
            free(stagingK_raw);
            free(stagingV_raw);
            return false;
        }
    }

    free(stagingK_raw);
    free(stagingV_raw);
    fsync(fd_);
    return true;
}

bool Streaming1MEngine::prepareSSDFileDirectFP16(const __fp16* srcK, const __fp16* srcV, size_t totalTokens) {
    return prepareSSDFileChunkedFP16(totalTokens, [&](size_t chunkStart, size_t chunkLen, __fp16* dstK, __fp16* dstV) {
        for (uint32_t h = 0; h < H_; h++) {
            const __fp16* k_src = srcK + ((size_t)h * totalTokens + chunkStart) * D_;
            const __fp16* v_src = srcV + ((size_t)h * totalTokens + chunkStart) * D_;
            __fp16* k_dst = dstK + ((size_t)h * slotTokens_) * D_;
            __fp16* v_dst = dstV + ((size_t)h * slotTokens_) * D_;

            memcpy(k_dst, k_src, chunkLen * D_ * sizeof(__fp16));
            memcpy(v_dst, v_src, chunkLen * D_ * sizeof(__fp16));
        }
    });
}

bool Streaming1MEngine::prepareSSDFileDirectQ8_0(const block_q8_0* srcK_q8, const block_q8_0* srcV_q8, size_t totalTokens) {
    size_t blocksPerToken = D_ / 32;
    return prepareSSDFileChunkedQ8_0(totalTokens, [&](size_t chunkStart, size_t chunkLen, block_q8_0* dstK, block_q8_0* dstV) {
        for (uint32_t h = 0; h < H_; h++) {
            const block_q8_0* k_src = srcK_q8 + ((size_t)h * totalTokens + chunkStart) * blocksPerToken;
            const block_q8_0* v_src = srcV_q8 + ((size_t)h * totalTokens + chunkStart) * blocksPerToken;
            block_q8_0* k_dst = dstK + ((size_t)h * slotTokens_) * blocksPerToken;
            block_q8_0* v_dst = dstV + ((size_t)h * slotTokens_) * blocksPerToken;

            memcpy(k_dst, k_src, chunkLen * blocksPerToken * sizeof(block_q8_0));
            memcpy(v_dst, v_src, chunkLen * blocksPerToken * sizeof(block_q8_0));
        }
    });
}

void Streaming1MEngine::prefetchChunkAsync(uint32_t slotIdx, uint32_t chunkIdx, size_t chunkStartToken, size_t chunkLen) {
    RingSlot& slot = slots_[slotIdx % 2];
    slot.currentTokens = chunkLen;
    slot.chunkStartToken = chunkStartToken;
    slot.chunkIndex = chunkIdx;
    slot.isPrefetching = true;

    int fd = fd_;
    size_t bytesPerChunk = bytesPerChunk_;
    size_t maxChunkBytesK = maxChunkBytesK_;
    size_t readBytesK = maxChunkBytesK_;
    size_t readBytesV = maxChunkBytesV_;

    void* dstK = isQ8_0_ ? [slot.bufferK_q8 contents] : [slot.bufferK contents];
    void* dstV = isQ8_0_ ? [slot.bufferV_q8 contents] : [slot.bufferV contents];

    dispatch_async(ioQueue_, ^{
        auto t0 = std::chrono::high_resolution_clock::now();

        off_t offK = (off_t)chunkIdx * bytesPerChunk;
        off_t offV = (off_t)chunkIdx * bytesPerChunk + (off_t)maxChunkBytesK;

        ssize_t rk = pread(fd, dstK, readBytesK, offK);
        ssize_t rv = pread(fd, dstV, readBytesV, offV);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsedSec = std::chrono::duration<double>(t1 - t0).count();

        if (rk < 0 || rv < 0) {
            perror("[-] Direct pread() from NVMe SSD failed");
        }

        totalReadBytes_ += (double)(readBytesK + readBytesV);
        totalReadTimeSec_ += elapsedSec;

        dispatch_semaphore_signal(slot.readySem);
    });
}

void Streaming1MEngine::waitForSlot(uint32_t slotIdx) {
    RingSlot& slot = slots_[slotIdx % 2];
    if (slot.isPrefetching) {
        dispatch_semaphore_wait(slot.readySem, DISPATCH_TIME_FOREVER);
        slot.isPrefetching = false;
    }
}

bool Streaming1MEngine::executeModeAStreamingAttention(
    id<MTLBuffer> d_Q,
    id<MTLBuffer> d_O,
    size_t totalTokens,
    size_t inRamActiveTokens,
    id<MTLBuffer> d_activeK,
    id<MTLBuffer> d_activeV,
    PerformanceMetrics* metricsOut)
{
    @autoreleasepool {
        double rss_start = get_current_rss_mb();
        resetMetrics();

        size_t ssdTokens = totalTokens > inRamActiveTokens ? (totalTokens - inRamActiveTokens) : 0;
        size_t numChunks = (ssdTokens + slotTokens_ - 1) / slotTokens_;
        bool hasActiveRAM = (inRamActiveTokens > 0 && d_activeK && d_activeV);
        size_t totalPasses = numChunks + (hasActiveRAM ? 1 : 0);

        id<MTLComputePipelineState> pso = isQ8_0_ ? psoStreamQ8_0_ : psoStreamFP16_;
        float scale = 1.0f / std::sqrt((float)D_);
        size_t smem_bytes = (D_ == 64) ? (4 * 64 * 64 * sizeof(__fp16)) : (4 * 64 * 128 * sizeof(__fp16));

        // State Buffers for Online Softmax Multi-Chunk Continuity
        id<MTLBuffer> m_state = [device_ newBufferWithLength:(size_t)H_ * totalTokens * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> l_state = [device_ newBufferWithLength:(size_t)H_ * totalTokens * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> O_state = [device_ newBufferWithLength:(size_t)H_ * totalTokens * D_ * sizeof(__fp16) options:MTLResourceStorageModeShared];

        auto host_t0 = std::chrono::high_resolution_clock::now();
        __block CFTimeInterval first_gpu_start = 0;
        __block CFTimeInterval last_gpu_end = 0;

        // Prefetch Slot 0 if SSD chunks exist
        if (numChunks > 0) {
            prefetchChunkAsync(0, 0, 0, std::min(slotTokens_, ssdTokens));
        }

        for (size_t c = 0; c < numChunks; c++) {
            uint32_t slotIdx = c % 2;
            uint32_t nextSlotIdx = (c + 1) % 2;
            size_t chunkStart = c * slotTokens_;
            size_t chunkLen = std::min(slotTokens_, ssdTokens - chunkStart);

            waitForSlot(slotIdx);

            if (c + 1 < numChunks) {
                size_t nextStart = (c + 1) * slotTokens_;
                size_t nextLen = std::min(slotTokens_, ssdTokens - nextStart);
                prefetchChunkAsync(nextSlotIdx, (uint32_t)(c + 1), nextStart, nextLen);
            }

            uint32_t M_u = (uint32_t)totalTokens;
            uint32_t c_slot = (uint32_t)slotTokens_;
            uint32_t k_start_u = (uint32_t)chunkStart;
            uint32_t k_len_u = (uint32_t)chunkLen;
            uint32_t is_first = (c == 0 ? 1 : 0);
            uint32_t is_last = (c == totalPasses - 1 ? 1 : 0);

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:d_Q offset:0 atIndex:0];

            if (isQ8_0_) {
                [enc setBuffer:getSlot(slotIdx).bufferK_q8 offset:0 atIndex:1];
                [enc setBuffer:getSlot(slotIdx).bufferV_q8 offset:0 atIndex:2];
            } else {
                [enc setBuffer:getSlot(slotIdx).bufferK offset:0 atIndex:1];
                [enc setBuffer:getSlot(slotIdx).bufferV offset:0 atIndex:2];
            }

            [enc setBuffer:d_O offset:0 atIndex:3];
            [enc setBuffer:m_state offset:0 atIndex:4];
            [enc setBuffer:l_state offset:0 atIndex:5];
            [enc setBuffer:O_state offset:0 atIndex:6];
            [enc setBytes:&M_u length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&c_slot length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&k_start_u length:sizeof(uint32_t) atIndex:9];
            [enc setBytes:&k_len_u length:sizeof(uint32_t) atIndex:10];
            [enc setBytes:&is_first length:sizeof(uint32_t) atIndex:11];
            [enc setBytes:&is_last length:sizeof(uint32_t) atIndex:12];
            [enc setBytes:&scale length:sizeof(float) atIndex:13];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((totalTokens + 63) / 64, H_, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];

            if (c == 0) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    first_gpu_start = buf.GPUStartTime;
                }];
            }
            if (c == totalPasses - 1) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    last_gpu_end = buf.GPUEndTime;
                }];
            }

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Active In-RAM Window Pass (if present)
        if (hasActiveRAM) {
            uint32_t M_u = (uint32_t)totalTokens;
            uint32_t c_slot = (uint32_t)inRamActiveTokens;
            uint32_t k_start_u = (uint32_t)ssdTokens;
            uint32_t k_len_u = (uint32_t)inRamActiveTokens;
            uint32_t is_first = (numChunks == 0 ? 1 : 0);
            uint32_t is_last = 1;

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:d_Q offset:0 atIndex:0];
            [enc setBuffer:d_activeK offset:0 atIndex:1];
            [enc setBuffer:d_activeV offset:0 atIndex:2];
            [enc setBuffer:d_O offset:0 atIndex:3];
            [enc setBuffer:m_state offset:0 atIndex:4];
            [enc setBuffer:l_state offset:0 atIndex:5];
            [enc setBuffer:O_state offset:0 atIndex:6];
            [enc setBytes:&M_u length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&c_slot length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&k_start_u length:sizeof(uint32_t) atIndex:9];
            [enc setBytes:&k_len_u length:sizeof(uint32_t) atIndex:10];
            [enc setBytes:&is_first length:sizeof(uint32_t) atIndex:11];
            [enc setBytes:&is_last length:sizeof(uint32_t) atIndex:12];
            [enc setBytes:&scale length:sizeof(float) atIndex:13];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((totalTokens + 63) / 64, H_, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];

            if (numChunks == 0) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    first_gpu_start = buf.GPUStartTime;
                }];
            }
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                last_gpu_end = buf.GPUEndTime;
            }];

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto host_t1 = std::chrono::high_resolution_clock::now();
        double host_elapsed_ms = std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();
        double gpu_elapsed_ms = (last_gpu_end > first_gpu_start) ? ((last_gpu_end - first_gpu_start) * 1000.0) : host_elapsed_ms;

        double causal_flops = 2.0 * (double)totalTokens * (double)totalTokens * (double)H_ * (double)D_;
        double tflops = (causal_flops / 1e12) / (gpu_elapsed_ms / 1000.0);
        double peak_rss = std::max(rss_start, get_current_rss_mb());

        if (metricsOut) {
            metricsOut->latency_ms = host_elapsed_ms;
            metricsOut->gpu_only_ms = gpu_elapsed_ms;
            metricsOut->tflops = tflops;
            metricsOut->ssd_bandwidth_gbps = getEffectiveReadBandwidthGBps();
            metricsOut->peak_rss_mb = peak_rss;
            metricsOut->total_read_bytes = totalReadBytes_;
            metricsOut->total_read_time_sec = totalReadTimeSec_;
            metricsOut->throughput_tok_per_sec = (double)totalTokens / (host_elapsed_ms / 1000.0);
        }

        m_state = nil;
        l_state = nil;
        O_state = nil;
        return true;
    }
}

bool Streaming1MEngine::executeModeBSpeculativeVerification(
    id<MTLBuffer> d_Q_spec,
    id<MTLBuffer> d_O_spec,
    uint32_t K_spec,
    size_t pastTokens,
    size_t inRamActiveTokens,
    id<MTLBuffer> d_activeK,
    id<MTLBuffer> d_activeV,
    PerformanceMetrics* metricsOut)
{
    @autoreleasepool {
        double rss_start = get_current_rss_mb();
        resetMetrics();

        size_t totalTokens = pastTokens + K_spec;
        size_t ssdTokens = totalTokens > inRamActiveTokens ? (totalTokens - inRamActiveTokens) : 0;
        size_t numChunks = (ssdTokens + slotTokens_ - 1) / slotTokens_;
        bool hasActiveRAM = (inRamActiveTokens > 0 && d_activeK && d_activeV);
        size_t totalPasses = numChunks + (hasActiveRAM ? 1 : 0);

        id<MTLComputePipelineState> pso = isQ8_0_ ? psoSpecVerifyQ8_0_ : psoSpecVerifyFP16_;
        float scale = 1.0f / std::sqrt((float)D_);
        size_t smem_bytes = (D_ == 64) ? (4 * 64 * 64 * sizeof(__fp16)) : (4 * 64 * 128 * sizeof(__fp16));

        id<MTLBuffer> m_state = [device_ newBufferWithLength:(size_t)H_ * K_spec * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> l_state = [device_ newBufferWithLength:(size_t)H_ * K_spec * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> O_state = [device_ newBufferWithLength:(size_t)H_ * K_spec * D_ * sizeof(__fp16) options:MTLResourceStorageModeShared];

        auto host_t0 = std::chrono::high_resolution_clock::now();
        __block CFTimeInterval first_gpu_start = 0;
        __block CFTimeInterval last_gpu_end = 0;

        if (numChunks > 0) {
            prefetchChunkAsync(0, 0, 0, std::min(slotTokens_, ssdTokens));
        }

        for (size_t c = 0; c < numChunks; c++) {
            uint32_t slotIdx = c % 2;
            uint32_t nextSlotIdx = (c + 1) % 2;
            size_t chunkStart = c * slotTokens_;
            size_t chunkLen = std::min(slotTokens_, ssdTokens - chunkStart);

            waitForSlot(slotIdx);

            if (c + 1 < numChunks) {
                size_t nextStart = (c + 1) * slotTokens_;
                size_t nextLen = std::min(slotTokens_, ssdTokens - nextStart);
                prefetchChunkAsync(nextSlotIdx, (uint32_t)(c + 1), nextStart, nextLen);
            }

            uint32_t k_spec_u = K_spec;
            uint32_t m_past_u = (uint32_t)pastTokens;
            uint32_t c_slot = (uint32_t)slotTokens_;
            uint32_t k_start_u = (uint32_t)chunkStart;
            uint32_t k_len_u = (uint32_t)chunkLen;
            uint32_t is_first = (c == 0 ? 1 : 0);
            uint32_t is_last = (c == totalPasses - 1 ? 1 : 0);

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:d_Q_spec offset:0 atIndex:0];

            if (isQ8_0_) {
                [enc setBuffer:getSlot(slotIdx).bufferK_q8 offset:0 atIndex:1];
                [enc setBuffer:getSlot(slotIdx).bufferV_q8 offset:0 atIndex:2];
            } else {
                [enc setBuffer:getSlot(slotIdx).bufferK offset:0 atIndex:1];
                [enc setBuffer:getSlot(slotIdx).bufferV offset:0 atIndex:2];
            }

            [enc setBuffer:d_O_spec offset:0 atIndex:3];
            [enc setBuffer:m_state offset:0 atIndex:4];
            [enc setBuffer:l_state offset:0 atIndex:5];
            [enc setBuffer:O_state offset:0 atIndex:6];
            [enc setBytes:&k_spec_u length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&m_past_u length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&c_slot length:sizeof(uint32_t) atIndex:9];
            [enc setBytes:&k_start_u length:sizeof(uint32_t) atIndex:10];
            [enc setBytes:&k_len_u length:sizeof(uint32_t) atIndex:11];
            [enc setBytes:&is_first length:sizeof(uint32_t) atIndex:12];
            [enc setBytes:&is_last length:sizeof(uint32_t) atIndex:13];
            [enc setBytes:&scale length:sizeof(float) atIndex:14];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, H_, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];

            if (c == 0) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    first_gpu_start = buf.GPUStartTime;
                }];
            }
            if (c == totalPasses - 1) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    last_gpu_end = buf.GPUEndTime;
                }];
            }

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        if (hasActiveRAM) {
            uint32_t k_spec_u = K_spec;
            uint32_t m_past_u = (uint32_t)pastTokens;
            uint32_t c_slot = (uint32_t)inRamActiveTokens;
            uint32_t k_start_u = (uint32_t)ssdTokens;
            uint32_t k_len_u = (uint32_t)inRamActiveTokens;
            uint32_t is_first = (numChunks == 0 ? 1 : 0);
            uint32_t is_last = 1;

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:d_Q_spec offset:0 atIndex:0];
            [enc setBuffer:d_activeK offset:0 atIndex:1];
            [enc setBuffer:d_activeV offset:0 atIndex:2];
            [enc setBuffer:d_O_spec offset:0 atIndex:3];
            [enc setBuffer:m_state offset:0 atIndex:4];
            [enc setBuffer:l_state offset:0 atIndex:5];
            [enc setBuffer:O_state offset:0 atIndex:6];
            [enc setBytes:&k_spec_u length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&m_past_u length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&c_slot length:sizeof(uint32_t) atIndex:9];
            [enc setBytes:&k_start_u length:sizeof(uint32_t) atIndex:10];
            [enc setBytes:&k_len_u length:sizeof(uint32_t) atIndex:11];
            [enc setBytes:&is_first length:sizeof(uint32_t) atIndex:12];
            [enc setBytes:&is_last length:sizeof(uint32_t) atIndex:13];
            [enc setBytes:&scale length:sizeof(float) atIndex:14];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, H_, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];

            if (numChunks == 0) {
                [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                    first_gpu_start = buf.GPUStartTime;
                }];
            }
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> buf) {
                last_gpu_end = buf.GPUEndTime;
            }];

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto host_t1 = std::chrono::high_resolution_clock::now();
        double host_elapsed_ms = std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();
        double gpu_elapsed_ms = (last_gpu_end > first_gpu_start) ? ((last_gpu_end - first_gpu_start) * 1000.0) : host_elapsed_ms;

        // FLOPs for K_spec queries over totalTokens: 4.0 * K_spec * totalTokens * H * D
        double verify_flops = 4.0 * (double)K_spec * (double)totalTokens * (double)H_ * (double)D_;
        double tflops = (verify_flops / 1e12) / (gpu_elapsed_ms / 1000.0);
        double peak_rss = std::max(rss_start, get_current_rss_mb());

        if (metricsOut) {
            metricsOut->latency_ms = host_elapsed_ms;
            metricsOut->gpu_only_ms = gpu_elapsed_ms;
            metricsOut->tflops = tflops;
            metricsOut->ssd_bandwidth_gbps = getEffectiveReadBandwidthGBps();
            metricsOut->peak_rss_mb = peak_rss;
            metricsOut->total_read_bytes = totalReadBytes_;
            metricsOut->total_read_time_sec = totalReadTimeSec_;
            metricsOut->throughput_tok_per_sec = (double)K_spec / (host_elapsed_ms / 1000.0);
        }

        m_state = nil;
        l_state = nil;
        O_state = nil;
        return true;
    }
}
