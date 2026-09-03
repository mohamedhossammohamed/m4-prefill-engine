#pragma once

#include <cstddef>
#include <cstdint>
#include "core/memory/quant_types.h"

namespace core::metrology {

// Thread-safe, controllable Linear Congruential Generator (LCG)
class PRNG {
public:
    explicit PRNG(uint32_t seed = 1337);
    void seed(uint32_t s);
    uint32_t next_u32();
    float rand_uniform(float min_val = -1.0f, float max_val = 1.0f);
    float rand_uniform_01(); // [0.0, 1.0)
    uint32_t get_seed() const { return state_; }

private:
    uint32_t state_;
};

// Thread-local global PRNG functions (thread-safe, deterministic)
void prng_seed(uint32_t seed);
void set_prng_seed(uint32_t seed);
uint32_t prng_next_u32();
uint32_t get_prng_seed();
float prng_rand_uniform(float min_val = -1.0f, float max_val = 1.0f);
float rand_uniform(); // [0.0, 1.0)

// Synthetic activation generation
void generate_uniform_activations(__fp16* data, size_t count, float min_val = -1.0f, float max_val = 1.0f);
void generate_uniform_activations(float* data, size_t count, float min_val = -1.0f, float max_val = 1.0f);

void generate_gaussian_activations(__fp16* data, size_t count, float mean = 0.0f, float std = 1.0f);
void generate_gaussian_activations(float* data, size_t count, float mean = 0.0f, float std = 1.0f);

// Synthetic quantized weight generators
void generate_q4_0_weights(core::memory::block_q4_0* blocks, size_t num_blocks, float scale_factor = 0.04f);
void generate_q8_0_weights(core::memory::block_q8_0* blocks, size_t num_blocks);
void quantize_fp16_to_q8_0(const __fp16* src, core::memory::block_q8_0* dst, size_t num_elements);
void dequantize_q8_0_to_fp16(const core::memory::block_q8_0* src, __fp16* dst, size_t num_elements);

} // namespace core::metrology
