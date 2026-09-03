#include "core/metrology/telemetry_format.h"

#include <sstream>
#include <iomanip>
#include <cmath>

namespace core::metrology {

std::string format_time(double ms, int precision) {
    std::ostringstream ss;
    if (ms >= 1000.0) {
        ss << std::fixed << std::setprecision(precision) << (ms / 1000.0) << " s";
    } else {
        ss << std::fixed << std::setprecision(precision) << ms << " ms";
    }
    return ss.str();
}

std::string format_throughput(double tok_per_sec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << tok_per_sec << " tok/s";
    return ss.str();
}

std::string format_tflops(double tflops) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << tflops << " TFLOP/s";
    return ss.str();
}

std::string format_bandwidth(double gbps) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << gbps << " GB/s";
    return ss.str();
}

std::string format_memory(double mb) {
    std::ostringstream ss;
    if (mb >= 1024.0) {
        ss << std::fixed << std::setprecision(2) << (mb / 1024.0) << " GB";
    } else {
        ss << std::fixed << std::setprecision(2) << mb << " MB";
    }
    return ss.str();
}

std::string format_verification_status(
    size_t M,
    size_t verification_threshold,
    bool passed)
{
    if (M > verification_threshold) {
        return "[NOT VERIFIED — CPU gold infeasible at this scale]";
    }
    return passed ? "[PASSED]" : "[FAILED]";
}

std::string format_diff_telemetry(
    float max_diff,
    bool is_gated,
    const char* gated_reason)
{
    if (is_gated) {
        return std::string("N/A (") + (gated_reason ? gated_reason : "Gated") + ")";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << max_diff;
    return ss.str();
}

} // namespace core::metrology
