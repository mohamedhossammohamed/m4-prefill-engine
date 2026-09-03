#include "core/metrology/tripwires.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <iomanip>

namespace core::metrology {

bool verify_finite(const float* data, size_t count, const char* tensor_name) {
    (void)tensor_name;
    if (!data) return false;
    for (size_t i = 0; i < count; i++) {
        float v = data[i];
        if (std::isnan(v) || std::isinf(v)) {
            return false;
        }
    }
    return true;
}

bool verify_finite(const __fp16* data, size_t count, const char* tensor_name) {
    (void)tensor_name;
    if (!data) return false;
    for (size_t i = 0; i < count; i++) {
        float v = static_cast<float>(data[i]);
        if (std::isnan(v) || std::isinf(v)) {
            return false;
        }
    }
    return true;
}

void assert_finite(const float* data, size_t count, const char* tensor_name) {
    if (!data) {
        fprintf(stderr, "\n[FATAL ERROR] Null pointer passed to assert_finite for %s\n", tensor_name ? tensor_name : "Tensor");
        assert(false && "Null pointer in assert_finite");
        exit(1);
    }
    for (size_t i = 0; i < count; i++) {
        float v = data[i];
        if (std::isnan(v) || std::isinf(v)) {
            fprintf(stderr, "\n[FATAL ERROR] NaN or Inf detected in %s at index %zu! Value: %f\n",
                    tensor_name ? tensor_name : "Tensor", i, v);
            assert(!std::isnan(v) && !std::isinf(v) && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
    }
}

void assert_finite(const __fp16* data, size_t count, const char* tensor_name) {
    if (!data) {
        fprintf(stderr, "\n[FATAL ERROR] Null pointer passed to assert_finite for %s\n", tensor_name ? tensor_name : "Tensor");
        assert(false && "Null pointer in assert_finite");
        exit(1);
    }
    for (size_t i = 0; i < count; i++) {
        float v = static_cast<float>(data[i]);
        if (std::isnan(v) || std::isinf(v)) {
            fprintf(stderr, "\n[FATAL ERROR] NaN or Inf detected in %s at index %zu! Value: %f\n",
                    tensor_name ? tensor_name : "Tensor", i, v);
            assert(!std::isnan(v) && !std::isinf(v) && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
    }
}

float compute_max_diff(const float* cpu_gold, const __fp16* gpu_out, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float va = static_cast<float>(gpu_out[i]);
        float vb = cpu_gold[i];
        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
            fprintf(stderr, "\n[FATAL ERROR] NaN or Inf detected during compute_max_diff at index %zu! GPU: %f | CPU: %f\n",
                    i, va, vb);
            assert(false && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
        float diff = std::fabs(va - vb);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

float compute_max_diff(const __fp16* cpu_gold, const __fp16* gpu_out, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float va = static_cast<float>(gpu_out[i]);
        float vb = static_cast<float>(cpu_gold[i]);
        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
            fprintf(stderr, "\n[FATAL ERROR] NaN or Inf detected during compute_max_diff at index %zu! GPU: %f | CPU: %f\n",
                    i, va, vb);
            assert(false && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
        float diff = std::fabs(va - vb);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

float compute_max_diff(const float* cpu_gold, const float* gpu_out, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float va = gpu_out[i];
        float vb = cpu_gold[i];
        if (std::isnan(va) || std::isnan(vb) || std::isinf(va) || std::isinf(vb)) {
            fprintf(stderr, "\n[FATAL ERROR] NaN or Inf detected during compute_max_diff at index %zu! GPU: %f | CPU: %f\n",
                    i, va, vb);
            assert(false && "Numerical instability detected (NaN/Inf)");
            exit(1);
        }
        float diff = std::fabs(va - vb);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

VerificationResult verify_tensor_parity(
    const __fp16* actual,
    const __fp16* expected,
    size_t count,
    float tolerance,
    const char* tensor_name)
{
    VerificationResult res;
    if (count == 0) {
        res.passed = true;
        res.details = "Empty tensor, trivially verified.";
        return res;
    }

    double sum_diff = 0.0;
    double sum_sq_diff = 0.0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < count; i++) {
        float va = static_cast<float>(actual[i]);
        float ve = static_cast<float>(expected[i]);

        if (std::isnan(va) || std::isnan(ve) || std::isinf(va) || std::isinf(ve)) {
            res.has_non_finite = true;
            res.passed = false;
            std::ostringstream ss;
            ss << "Non-finite value detected at index " << i << " in " << (tensor_name ? tensor_name : "Tensor")
               << " (actual=" << va << ", expected=" << ve << ")";
            res.details = ss.str();
            return res;
        }

        float diff = std::fabs(va - ve);
        if (diff > max_diff) {
            max_diff = diff;
        }
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }

    res.max_diff = max_diff;
    res.avg_diff = static_cast<float>(sum_diff / count);
    res.rmse = static_cast<float>(std::sqrt(sum_sq_diff / count));
    res.passed = (max_diff <= tolerance);

    std::ostringstream ss;
    ss << (tensor_name ? tensor_name : "Tensor")
       << " Parity: MaxDiff=" << std::fixed << std::setprecision(6) << res.max_diff
       << ", AvgDiff=" << res.avg_diff
       << ", RMSE=" << res.rmse
       << (res.passed ? " [PASSED]" : " [FAILED tolerance violation]");
    res.details = ss.str();
    return res;
}

VerificationResult verify_tensor_parity(
    const __fp16* actual,
    const float* expected,
    size_t count,
    float tolerance,
    const char* tensor_name)
{
    VerificationResult res;
    if (count == 0) {
        res.passed = true;
        res.details = "Empty tensor, trivially verified.";
        return res;
    }

    double sum_diff = 0.0;
    double sum_sq_diff = 0.0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < count; i++) {
        float va = static_cast<float>(actual[i]);
        float ve = expected[i];

        if (std::isnan(va) || std::isnan(ve) || std::isinf(va) || std::isinf(ve)) {
            res.has_non_finite = true;
            res.passed = false;
            std::ostringstream ss;
            ss << "Non-finite value detected at index " << i << " in " << (tensor_name ? tensor_name : "Tensor")
               << " (actual=" << va << ", expected=" << ve << ")";
            res.details = ss.str();
            return res;
        }

        float diff = std::fabs(va - ve);
        if (diff > max_diff) {
            max_diff = diff;
        }
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }

    res.max_diff = max_diff;
    res.avg_diff = static_cast<float>(sum_diff / count);
    res.rmse = static_cast<float>(std::sqrt(sum_sq_diff / count));
    res.passed = (max_diff <= tolerance);

    std::ostringstream ss;
    ss << (tensor_name ? tensor_name : "Tensor")
       << " Parity: MaxDiff=" << std::fixed << std::setprecision(6) << res.max_diff
       << ", AvgDiff=" << res.avg_diff
       << ", RMSE=" << res.rmse
       << (res.passed ? " [PASSED]" : " [FAILED tolerance violation]");
    res.details = ss.str();
    return res;
}

void check_drift_assertion(float diff, float threshold, const char* test_name) {
    if (diff > threshold) {
        fprintf(stderr, "\n[FATAL ERROR] Drift threshold exceeded in %s! Diff: %f, Threshold: %f\n",
                test_name ? test_name : "Drift Test", diff, threshold);
        assert(diff <= threshold && "Thermal / drift stability assertion failed!");
        exit(1);
    }
}

} // namespace core::metrology
