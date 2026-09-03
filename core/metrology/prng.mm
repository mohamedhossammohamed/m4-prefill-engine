#include "core/metrology/prng.h"

#include <cmath>
#include <algorithm>

namespace core::metrology {

PRNG::PRNG(uint32_t seed) : state_(seed) {}

void PRNG::seed(uint32_t s) {
    state_ = s;
}

uint32_t PRNG::next_u32() {
    state_ = state_ * 1664525u + 1013904223u;
    return state_;
}

float PRNG::rand_uniform_01() {
    return static_cast<float>(next_u32()) / static_cast<float>(0xFFFFFFFF);
}

float PRNG::rand_uniform(float min_val, float max_val) {
    return min_val + rand_uniform_01() * (max_val - min_val);
}

static thread_local PRNG t_prng(1337);

void prng_seed(uint32_t seed) {
    t_prng.seed(seed);
}

void set_prng_seed(uint32_t seed) {
    t_prng.seed(seed);
}

uint32_t prng_next_u32() {
    return t_prng.next_u32();
}

uint32_t get_prng_seed() {
    return t_prng.get_seed();
}

float prng_rand_uniform(float min_val, float max_val) {
    return t_prng.rand_uniform(min_val, max_val);
}

float rand_uniform() {
    return t_prng.rand_uniform_01();
}

void generate_uniform_activations(__fp16* data, size_t count, float min_val, float max_val) {
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<__fp16>(prng_rand_uniform(min_val, max_val));
    }
}

void generate_uniform_activations(float* data, size_t count, float min_val, float max_val) {
    for (size_t i = 0; i < count; i++) {
        data[i] = prng_rand_uniform(min_val, max_val);
    }
}

void generate_gaussian_activations(__fp16* data, size_t count, float mean, float std) {
    for (size_t i = 0; i < count; i += 2) {
        float u1 = rand_uniform();
        float u2 = rand_uniform();
        if (u1 < 1e-6f) u1 = 1e-6f;
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * static_cast<float>(M_PI) * u2;
        float z0 = r * std::cos(theta);
        float z1 = r * std::sin(theta);
        data[i] = static_cast<__fp16>(mean + z0 * std::abs(std));
        if (i + 1 < count) {
            data[i + 1] = static_cast<__fp16>(mean + z1 * std::abs(std));
        }
    }
}

void generate_gaussian_activations(float* data, size_t count, float mean, float std) {
    for (size_t i = 0; i < count; i += 2) {
        float u1 = rand_uniform();
        float u2 = rand_uniform();
        if (u1 < 1e-6f) u1 = 1e-6f;
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * static_cast<float>(M_PI) * u2;
        float z0 = r * std::cos(theta);
        float z1 = r * std::sin(theta);
        data[i] = mean + z0 * std::abs(std);
        if (i + 1 < count) {
            data[i + 1] = mean + z1 * std::abs(std);
        }
    }
}

void generate_q4_0_weights(core::memory::block_q4_0* blocks, size_t num_blocks, float scale_factor) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = static_cast<__fp16>(rand_uniform() * scale_factor + scale_factor * 0.1f);
        for (int i = 0; i < 16; i++) {
            uint8_t low = static_cast<uint8_t>(rand_uniform() * 16.0f);
            uint8_t high = static_cast<uint8_t>(rand_uniform() * 16.0f);
            blocks[b].qs[i] = (high << 4) | (low & 0x0F);
        }
    }
}

void generate_q8_0_weights(core::memory::block_q8_0* blocks, size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; b++) {
        blocks[b].d = static_cast<__fp16>(rand_uniform() * 0.05f + 0.005f);
        for (int i = 0; i < 32; i++) {
            blocks[b].qs[i] = static_cast<int8_t>(prng_rand_uniform(-127.0f, 127.0f));
        }
    }
}

void quantize_fp16_to_q8_0(const __fp16* src, core::memory::block_q8_0* dst, size_t num_elements) {
    const size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; b++) {
        const __fp16* s = src + b * 32;
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            float v = std::abs(static_cast<float>(s[i]));
            if (v > max_abs) max_abs = v;
        }
        float d = max_abs / 127.0f;
        dst[b].d = static_cast<__fp16>(d);
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        for (int i = 0; i < 32; i++) {
            float v = static_cast<float>(s[i]) * id;
            dst[b].qs[i] = static_cast<int8_t>(std::round(v));
        }
    }
}

void dequantize_q8_0_to_fp16(const core::memory::block_q8_0* src, __fp16* dst, size_t num_elements) {
    const size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; b++) {
        float d = static_cast<float>(src[b].d);
        __fp16* d_out = dst + b * 32;
        for (int i = 0; i < 32; i++) {
            d_out[i] = static_cast<__fp16>(src[b].qs[i] * d);
        }
    }
}

} // namespace core::metrology
