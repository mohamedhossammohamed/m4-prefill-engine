#include "core/metrology/telemetry.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace core::metrology {

double compute_median(std::vector<double>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    if (n % 2 == 1) {
        return samples[n / 2];
    }
    return (samples[n / 2 - 1] + samples[n / 2]) * 0.5;
}

double compute_median(const std::vector<double>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    std::vector<double> copy = samples;
    return compute_median(copy);
}

double compute_variance_percentage(const std::vector<double>& samples, double baseline_median) {
    if (samples.empty() || baseline_median == 0.0) {
        return 0.0;
    }
    double cur_median = compute_median(samples);
    return compute_variance_percentage(cur_median, baseline_median);
}

double compute_variance_percentage(double current_median, double baseline_median) {
    if (baseline_median == 0.0) {
        return 0.0;
    }
    return (std::abs(current_median - baseline_median) / baseline_median) * 100.0;
}

SampleStats compute_sample_stats(const std::vector<double>& samples) {
    SampleStats stats;
    if (samples.empty()) {
        return stats;
    }
    std::vector<double> copy = samples;
    std::sort(copy.begin(), copy.end());
    size_t n = copy.size();

    stats.min_val = copy.front();
    stats.max_val = copy.back();
    stats.median = (n % 2 == 1) ? copy[n / 2] : ((copy[n / 2 - 1] + copy[n / 2]) * 0.5);

    double sum = std::accumulate(copy.begin(), copy.end(), 0.0);
    stats.mean = sum / static_cast<double>(n);

    double sum_sq_diff = 0.0;
    for (double v : copy) {
        double diff = v - stats.mean;
        sum_sq_diff += diff * diff;
    }
    stats.std_dev = (n > 1) ? std::sqrt(sum_sq_diff / static_cast<double>(n - 1)) : 0.0;

    return stats;
}

BenchmarkTimer::BenchmarkTimer()
    : wall_running_(false), gpu_compute_ms_(0.0), io_ms_(0.0) {}

void BenchmarkTimer::start_wall() {
    wall_start_ = std::chrono::high_resolution_clock::now();
    wall_running_ = true;
}

void BenchmarkTimer::stop_wall() {
    wall_end_ = std::chrono::high_resolution_clock::now();
    wall_running_ = false;
}

void BenchmarkTimer::record_gpu_command_buffer(id<MTLCommandBuffer> cmdBuffer) {
    if (!cmdBuffer) {
        gpu_compute_ms_ = 0.0;
        return;
    }
    if (cmdBuffer.status == MTLCommandBufferStatusCompleted) {
        double duration_sec = cmdBuffer.GPUEndTime - cmdBuffer.GPUStartTime;
        gpu_compute_ms_ = (duration_sec > 0.0) ? (duration_sec * 1000.0) : 0.0;
    }
}

void BenchmarkTimer::set_io_duration_ms(double ms) {
    io_ms_ = ms;
}

ExecutionTiming BenchmarkTimer::get_timing() const {
    ExecutionTiming timing;
    auto end_time = wall_running_ ? std::chrono::high_resolution_clock::now() : wall_end_;
    std::chrono::duration<double, std::milli> elapsed = end_time - wall_start_;
    timing.wall_ms = elapsed.count();
    timing.gpu_compute_ms = gpu_compute_ms_;
    timing.io_ms = io_ms_;
    return timing;
}

} // namespace core::metrology
