#pragma once

#include <string>
#include <cstddef>

namespace core::metrology {

// Formats execution duration into human-readable cognitive units (Invariant 11)
// Values >= 1000.0 ms are displayed in seconds (e.g. "19.60 s"); values < 1000.0 ms in "ms".
std::string format_time(double ms, int precision = 2);

// Formats throughput (e.g. "12,450.2 tok/s")
std::string format_throughput(double tok_per_sec);

// Formats computational throughput in TFLOP/s (e.g. "45.21 TFLOP/s")
std::string format_tflops(double tflops);

// Formats memory or I/O bandwidth in GB/s (e.g. "185.3 GB/s")
std::string format_bandwidth(double gbps);

// Formats memory consumption in MB or GB
std::string format_memory(double mb);

// Generates honest verification status string strictly enforcing Invariant 3:
// For M <= verification_threshold (default 2048): returns "[PASSED]" or "[FAILED]".
// For M > verification_threshold: returns "[NOT VERIFIED — CPU gold infeasible at this scale]".
// NEVER returns deceptive "[LOCKED]" on unverified rows.
std::string format_verification_status(
    size_t M,
    size_t verification_threshold = 2048,
    bool passed = true
);

// Generates honest max difference string strictly enforcing Invariant 9:
// When CPU gold reference is gated (e.g. M > 128): returns "N/A (CPU Gold Reference Gated for M > 128)".
// Otherwise returns actual calculated floating point difference.
// NEVER outputs fabricated float literals ("0.000000") or misleading "N/A (GPU-Only)".
std::string format_diff_telemetry(
    float max_diff,
    bool is_gated = false,
    const char* gated_reason = "CPU Gold Reference Gated for M > 128"
);

} // namespace core::metrology
