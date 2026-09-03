#pragma once

#include <cstddef>

namespace core::memory {

// Queries Mach kernel for exact physical footprint of process in bytes,
// capturing unified Metal GPU buffer allocations (MTLResourceStorageModeShared).
// Strictly enforces Invariant 4 (replaces mach_task_basic_info.resident_size).
size_t get_uma_phys_footprint_bytes();

// Returns physical footprint in megabytes (MB)
double get_uma_phys_footprint_mb();

// RAII helper to monitor physical footprint baseline, peak, and net growth
class ScopedUMATracker {
public:
    ScopedUMATracker();
    void sample();
    double baseline_mb() const noexcept { return baseline_mb_; }
    double peak_mb() const noexcept { return peak_mb_; }
    double current_mb() const;
    double net_growth_mb() const;
    void reset();

private:
    double baseline_mb_;
    double peak_mb_;
};

} // namespace core::memory
