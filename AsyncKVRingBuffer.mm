#import "AsyncKVRingBuffer.h"

AsyncKVRingBuffer::AsyncKVRingBuffer(
    id<MTLDevice> device,
    uint32_t H,
    uint32_t D,
    size_t slotTokens,
    bool isQ8_0)
    : device_(device),
      H_(H),
      D_(D),
      slotTokens_(slotTokens),
      isQ8_0_(isQ8_0),
      fd_(-1),
      totalReadBytes_(0.0),
      totalReadTimeSec_(0.0)
{
    ioQueue_ = dispatch_queue_create("com.m4engine.kv_streaming_io", DISPATCH_QUEUE_SERIAL);

    if (isQ8_0_) {
        // Q8_0: 34 bytes per 32 elements -> (D / 32) * sizeof(block_q8_0) per token
        size_t blocksPerToken = D_ / 32;
        bytesPerTokenK_ = H_ * blocksPerToken * sizeof(block_q8_0);
        bytesPerTokenV_ = H_ * blocksPerToken * sizeof(block_q8_0);
    } else {
        // FP16: 2 bytes per element -> H * D * 2 bytes per token
        bytesPerTokenK_ = H_ * D_ * sizeof(__fp16);
        bytesPerTokenV_ = H_ * D_ * sizeof(__fp16);
    }

    maxChunkBytesK_ = slotTokens_ * bytesPerTokenK_;
    maxChunkBytesV_ = slotTokens_ * bytesPerTokenV_;
    bytesPerChunk_  = maxChunkBytesK_ + maxChunkBytesV_;

    // Allocate Dual Unified Memory Ring Buffer Slots (Shared Memory Mode for direct CPU/GPU access)
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

AsyncKVRingBuffer::~AsyncKVRingBuffer() {
    if (fd_ >= 0) {
        ftruncate(fd_, 0); // Immediately release APFS block extents
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

bool AsyncKVRingBuffer::prepareSSDFileFP16(const __fp16* srcK, const __fp16* srcV, size_t totalTokens) {
    if (fd_ >= 0) {
        ftruncate(fd_, 0);
        close(fd_);
        fd_ = -1;
    }

    char tempPath[PATH_MAX];
    snprintf(tempPath, sizeof(tempPath), "/tmp/streaming_kv_fp16_%d_%p.bin", getpid(), this);
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
    std::vector<__fp16> stagingK(H_ * slotTokens_ * D_);
    std::vector<__fp16> stagingV(H_ * slotTokens_ * D_);

    for (size_t c = 0; c < numChunks; c++) {
        size_t chunkStart = c * slotTokens_;
        size_t chunkLen = std::min(slotTokens_, totalTokens - chunkStart);

        // Format chunk as [H, chunkLen, D] for coalesced GPU access
        for (uint32_t h = 0; h < H_; h++) {
            const __fp16* k_src = srcK + ((size_t)h * totalTokens + chunkStart) * D_;
            const __fp16* v_src = srcV + ((size_t)h * totalTokens + chunkStart) * D_;
            __fp16* k_dst = stagingK.data() + ((size_t)h * slotTokens_) * D_;
            __fp16* v_dst = stagingV.data() + ((size_t)h * slotTokens_) * D_;

            memcpy(k_dst, k_src, chunkLen * D_ * sizeof(__fp16));
            memcpy(v_dst, v_src, chunkLen * D_ * sizeof(__fp16));
        }

        off_t offK = (off_t)c * bytesPerChunk_;
        off_t offV = (off_t)c * bytesPerChunk_ + (off_t)maxChunkBytesK_;

        size_t writeBytesK = H_ * slotTokens_ * D_ * sizeof(__fp16);
        size_t writeBytesV = H_ * slotTokens_ * D_ * sizeof(__fp16);

        ssize_t wk = pwrite(fd_, stagingK.data(), writeBytesK, offK);
        ssize_t wv = pwrite(fd_, stagingV.data(), writeBytesV, offV);

        if (wk != (ssize_t)writeBytesK || wv != (ssize_t)writeBytesV) {
            perror("[-] Failed to write chunk to NVMe SSD file");
            return false;
        }
    }

    // Flush written data to physical NVMe NAND flash
    fsync(fd_);
    return true;
}

bool AsyncKVRingBuffer::prepareSSDFileQ8_0(const block_q8_0* srcK_q8, const block_q8_0* srcV_q8, size_t totalTokens) {
    if (fd_ >= 0) {
        ftruncate(fd_, 0);
        close(fd_);
        fd_ = -1;
    }

    char tempPath[PATH_MAX];
    snprintf(tempPath, sizeof(tempPath), "/tmp/streaming_kv_q8_0_%d_%p.bin", getpid(), this);
    tempFilePath_ = tempPath;

    fd_ = open(tempFilePath_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) {
        perror("[-] Failed to open temporary NVMe SSD file for Q8_0 KV cache");
        return false;
    }

    // STRICT INVARIANT 4: Unlink immediately after open so macOS guarantees zero disk litter
    unlink(tempFilePath_.c_str());

    // Disable macOS Unified Buffer Cache to force direct hardware NVMe I/O
    if (fcntl(fd_, F_NOCACHE, 1) < 0) {
        perror("[-] Warning: Failed to set F_NOCACHE direct I/O on SSD file");
    }

    size_t blocksPerToken = D_ / 32;
    size_t numChunks = (totalTokens + slotTokens_ - 1) / slotTokens_;
    std::vector<block_q8_0> stagingK(H_ * slotTokens_ * blocksPerToken);
    std::vector<block_q8_0> stagingV(H_ * slotTokens_ * blocksPerToken);

    for (size_t c = 0; c < numChunks; c++) {
        size_t chunkStart = c * slotTokens_;
        size_t chunkLen = std::min(slotTokens_, totalTokens - chunkStart);

        // Format chunk as [H, chunkLen, D/32 blocks]
        for (uint32_t h = 0; h < H_; h++) {
            const block_q8_0* k_src = srcK_q8 + ((size_t)h * totalTokens + chunkStart) * blocksPerToken;
            const block_q8_0* v_src = srcV_q8 + ((size_t)h * totalTokens + chunkStart) * blocksPerToken;
            block_q8_0* k_dst = stagingK.data() + ((size_t)h * slotTokens_) * blocksPerToken;
            block_q8_0* v_dst = stagingV.data() + ((size_t)h * slotTokens_) * blocksPerToken;

            memcpy(k_dst, k_src, chunkLen * blocksPerToken * sizeof(block_q8_0));
            memcpy(v_dst, v_src, chunkLen * blocksPerToken * sizeof(block_q8_0));
        }

        off_t offK = (off_t)c * bytesPerChunk_;
        off_t offV = (off_t)c * bytesPerChunk_ + (off_t)maxChunkBytesK_;

        size_t writeBytesK = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);
        size_t writeBytesV = H_ * slotTokens_ * blocksPerToken * sizeof(block_q8_0);

        ssize_t wk = pwrite(fd_, stagingK.data(), writeBytesK, offK);
        ssize_t wv = pwrite(fd_, stagingV.data(), writeBytesV, offV);

        if (wk != (ssize_t)writeBytesK || wv != (ssize_t)writeBytesV) {
            perror("[-] Failed to write Q8_0 chunk to NVMe SSD file");
            return false;
        }
    }

    fsync(fd_);
    return true;
}

void AsyncKVRingBuffer::prefetchChunkAsync(uint32_t slotIdx, uint32_t chunkIdx, size_t chunkStartToken, size_t chunkLen) {
    Slot& slot = slots_[slotIdx % 2];
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

void AsyncKVRingBuffer::waitForSlot(uint32_t slotIdx) {
    Slot& slot = slots_[slotIdx % 2];
    if (slot.isPrefetching) {
        dispatch_semaphore_wait(slot.readySem, DISPATCH_TIME_FOREVER);
        slot.isPrefetching = false;
    }
}
