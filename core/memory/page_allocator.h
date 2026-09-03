#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <new>
#include <utility>

namespace core::memory {

// macOS ARM64 Apple Silicon system page size for F_NOCACHE Direct I/O
constexpr size_t SYSTEM_PAGE_SIZE_16KB = 16384;

// Checks whether pointer is aligned to 16KB page boundary
inline bool is_16kb_aligned(const void* ptr) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) % SYSTEM_PAGE_SIZE_16KB) == 0;
}

// Programmatic invariant assertion: verifies 16KB system page alignment
inline void assert_16kb_aligned(const void* ptr) {
    assert(is_16kb_aligned(ptr) && "Buffer MUST be 16KB aligned for F_NOCACHE Direct I/O!");
}

// Allocates 16KB-aligned memory block using posix_memalign
void* allocate_16kb_aligned(size_t bytes);

// Releases 16KB-aligned memory block
void free_16kb_aligned(void* ptr);

// RAII container managing 16KB-aligned heap memory for Direct I/O
template <typename T = uint8_t>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept
        : ptr_(nullptr), count_(0), bytes_(0), alignment_(SYSTEM_PAGE_SIZE_16KB) {}

    explicit AlignedBuffer(size_t count, size_t alignment = SYSTEM_PAGE_SIZE_16KB)
        : ptr_(nullptr), count_(0), bytes_(0), alignment_(alignment) {
        allocate(count, alignment);
    }

    ~AlignedBuffer() {
        release();
    }

    // Disable copy semantics to prevent double frees
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Enable move semantics
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), bytes_(other.bytes_), alignment_(other.alignment_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            bytes_ = other.bytes_;
            alignment_ = other.alignment_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.bytes_ = 0;
        }
        return *this;
    }

    void resize(size_t count, size_t alignment = SYSTEM_PAGE_SIZE_16KB) {
        if (count == count_ && alignment == alignment_) {
            return;
        }
        release();
        allocate(count, alignment);
    }

    void reset() noexcept {
        release();
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    size_t size() const noexcept { return count_; }
    size_t count() const noexcept { return count_; }
    size_t bytes() const noexcept { return bytes_; }
    size_t alignment() const noexcept { return alignment_; }
    bool empty() const noexcept { return count_ == 0; }

    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }

    T* begin() noexcept { return ptr_; }
    const T* begin() const noexcept { return ptr_; }
    T* end() noexcept { return ptr_ + count_; }
    const T* end() const noexcept { return ptr_ + count_; }

private:
    void allocate(size_t count, size_t alignment) {
        if (count == 0) {
            ptr_ = nullptr;
            count_ = 0;
            bytes_ = 0;
            alignment_ = alignment;
            return;
        }
        alignment_ = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;
        count_ = count;
        bytes_ = count * sizeof(T);

        // Ensure total bytes allocated is at least aligned or rounded up if desired
        void* raw_ptr = nullptr;
        int res = posix_memalign(&raw_ptr, alignment_, bytes_);
        if (res != 0 || raw_ptr == nullptr) {
            throw std::bad_alloc();
        }
        if (alignment_ == SYSTEM_PAGE_SIZE_16KB) {
            assert_16kb_aligned(raw_ptr);
        }
        ptr_ = static_cast<T*>(raw_ptr);
    }

    void release() noexcept {
        if (ptr_) {
            std::free(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
        bytes_ = 0;
    }

    T* ptr_;
    size_t count_;
    size_t bytes_;
    size_t alignment_;
};

} // namespace core::memory
