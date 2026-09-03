#pragma once

#include <vector>
#include <chrono>
#import <Metal/Metal.h>

namespace core::metrology {

struct SampleStats {
    double median = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double mean = 0.0;
    double std_dev = 0.0;
};

struct ExecutionTiming {
    double wall_ms = 0.0;
    double gpu_compute_ms = 0.0;
    double io_ms = 0.0;
};

// Computes median of execution time samples across N runs
double compute_median(std::vector<double>& samples);
double compute_median(const std::vector<double>& samples);

// Computes variance percentage of samples against baseline median
// (|median - baseline| / baseline) * 100.0
double compute_variance_percentage(const std::vector<double>& samples, double baseline_median);
double compute_variance_percentage(double current_median, double baseline_median);

// Full distribution summary statistics
SampleStats compute_sample_stats(const std::vector<double>& samples);

// High-precision timer capturing host wall-clock alongside GPU command buffer execution
// strictly enforcing Invariant 5 (Dual E2E vs Compute Tracking)
class BenchmarkTimer {
public:
    BenchmarkTimer();
    void start_wall();
    void stop_wall();
    void record_gpu_command_buffer(id<MTLCommandBuffer> cmdBuffer);
    void set_io_duration_ms(double ms);
    ExecutionTiming get_timing() const;

private:
    std::chrono::high_resolution_clock::time_point wall_start_;
    std::chrono::high_resolution_clock::time_point wall_end_;
    bool wall_running_;
    double gpu_compute_ms_;
    double io_ms_;
};

} // namespace core::metrology
