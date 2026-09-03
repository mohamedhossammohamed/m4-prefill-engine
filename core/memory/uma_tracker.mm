#include "core/memory/uma_tracker.h"

#include <mach/mach.h>
#include <mach/task_info.h>
#include <algorithm>

namespace core::memory {

size_t get_uma_phys_footprint_bytes() {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &count);
    if (kr != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(vm_info.phys_footprint);
}

double get_uma_phys_footprint_mb() {
    return static_cast<double>(get_uma_phys_footprint_bytes()) / (1024.0 * 1024.0);
}

ScopedUMATracker::ScopedUMATracker() {
    reset();
}

void ScopedUMATracker::reset() {
    baseline_mb_ = get_uma_phys_footprint_mb();
    peak_mb_ = baseline_mb_;
}

void ScopedUMATracker::sample() {
    double cur = get_uma_phys_footprint_mb();
    if (cur > peak_mb_) {
        peak_mb_ = cur;
    }
}

double ScopedUMATracker::current_mb() const {
    return get_uma_phys_footprint_mb();
}

double ScopedUMATracker::net_growth_mb() const {
    return peak_mb_ - baseline_mb_;
}

} // namespace core::memory
