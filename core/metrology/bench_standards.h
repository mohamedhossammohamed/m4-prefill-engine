#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

namespace core::metrology {

// Standard benchmark iteration counts (Original Request & Invariants)
constexpr size_t BENCHMARK_ITERATIONS = 20;
constexpr size_t BENCHMARK_WARMUP_RUNS = 3;
constexpr size_t BENCHMARK_COLD_RUNS = 5;

// Mandatory boundary token test counts for numerical verification
// Boundary counts M in {33, 127, 128, 129, 2048}
// Odd counts exercise partial threadgroup tiles; 128 & 2048 exercise full power-of-2 tiles.
constexpr size_t NUM_BOUNDARY_TOKEN_COUNTS = 5;
constexpr size_t BOUNDARY_TOKEN_COUNTS[NUM_BOUNDARY_TOKEN_COUNTS] = {33, 127, 128, 129, 2048};

inline bool is_boundary_token_count(size_t M) noexcept {
    for (size_t count : BOUNDARY_TOKEN_COUNTS) {
        if (M == count) return true;
    }
    return false;
}

// Numerical Parity Tolerances
constexpr float MAX_FP16_PARITY_DIFF = 0.0078125f;  // 1/128 (IEEE-754 half precision threshold)
constexpr float MAX_RELAXED_PARITY_DIFF = 0.05f;    // Relaxed threshold for long cumulative sequence reductions

// Hardware Occupancy & Telemetry Envelope Constraints
constexpr double MAX_LATENCY_VARIANCE_PCT = 1.5;     // 1.5% variance envelope across 20 iterations
constexpr uint32_t MAX_REGISTERS_PER_THREAD = 64;   // <= 64 registers per thread (100% occupancy)
constexpr size_t SRAM_PADDED_STRIDE_36 = 36;         // Stride 36 (32 + 4 pad) to avoid bank conflicts (Invariant 8)

// Invariant 10: Mandatory Apples-to-Apples Benchmarking Disclosure
inline const char* get_benchmark_disclosure_block() {
    return "================================================================================\n"
           " METROLOGY DISCLOSURE (Invariant 10: Apples-to-Apples Benchmarking Standards)\n"
           " - Format Parity: MLX 4-bit vs MLX 4-bit | Q4_0 vs Q4_0\n"
           " - Context Boundary: In-core prefill (M <= 2048), synthetic in-UMA weights, zero disk I/O\n"
           " - Verification Gate: M <= 2048 CPU gold verified; M > 2048 labeled NOT VERIFIED\n"
           " - Cold Eviction: 32MB Direct I/O UBC purge + 32MB SLC cache flush per cold run\n"
           " - Memory Accounting: task_vm_info.phys_footprint (capturing Metal UMA allocations)\n"
           "================================================================================\n";
}

} // namespace core::metrology
