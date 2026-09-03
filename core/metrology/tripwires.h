#pragma once

#include <cstddef>
#include <string>

namespace core::metrology {

struct VerificationResult {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    float rmse = 0.0f;
    bool passed = false;
    bool has_non_finite = false;
    std::string details;
};

// Non-finite (NaN/Inf) verification functions (Invariant 6)
// Returns false immediately if any element is NaN or Inf.
bool verify_finite(const float* data, size_t count, const char* tensor_name = "Tensor");
bool verify_finite(const __fp16* data, size_t count, const char* tensor_name = "Tensor");

// Non-finite tripwire assertions: aborts immediately on NaN/Inf with diagnostics
void assert_finite(const float* data, size_t count, const char* tensor_name = "Tensor");
void assert_finite(const __fp16* data, size_t count, const char* tensor_name = "Tensor");

// Computes maximum absolute difference between CPU gold reference and GPU output
float compute_max_diff(const float* cpu_gold, const __fp16* gpu_out, size_t count);
float compute_max_diff(const __fp16* cpu_gold, const __fp16* gpu_out, size_t count);
float compute_max_diff(const float* cpu_gold, const float* gpu_out, size_t count);

// Comprehensive tensor parity verification checking non-finite tripwires and max diff
VerificationResult verify_tensor_parity(
    const __fp16* actual,
    const __fp16* expected,
    size_t count,
    float tolerance = 0.05f,
    const char* tensor_name = "Output"
);

VerificationResult verify_tensor_parity(
    const __fp16* actual,
    const float* expected,
    size_t count,
    float tolerance = 0.05f,
    const char* tensor_name = "Output"
);

// Drift assertion check for stability / thermal testing
void check_drift_assertion(float diff, float threshold = 0.05f, const char* test_name = "Thermal Test");

} // namespace core::metrology
