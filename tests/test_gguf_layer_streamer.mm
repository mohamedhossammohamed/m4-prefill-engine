// GGUFLayerStreamer verification: bounded residency + bit-identical output.
// Builds a synthetic multi-layer Q2_0 GGUF fixture, serves it with
// resident_layer_count=1, and asserts:
//   (a) peak resident bytes stay bounded as total model bytes grow, and
//   (b) streamed bytes are bit-identical to the full-resident loader path.

#include "core/weights/gguf_layer_streamer.h"
#include "core/weights/gguf_loader.h"
#include "core/memory/quant_types.h"
#include "core/memory/uma_tracker.h"
#include "tests/e2e/test_common.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>

using namespace metal_llm;

static void append_le32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back((uint8_t)(v & 0xFF));
    buf.push_back((uint8_t)((v >> 8) & 0xFF));
    buf.push_back((uint8_t)((v >> 16) & 0xFF));
    buf.push_back((uint8_t)((v >> 24) & 0xFF));
}

static void append_le64(std::vector<uint8_t>& buf, uint64_t v) {
    append_le32(buf, (uint32_t)(v & 0xFFFFFFFF));
    append_le32(buf, (uint32_t)((v >> 32) & 0xFFFFFFFF));
}

static void append_string(std::vector<uint8_t>& buf, const std::string& str) {
    append_le64(buf, (uint64_t)str.size());
    buf.insert(buf.end(), str.begin(), str.end());
}

// Multi-layer fixture: `num_layers` x 2 Q2_0 tensors + 1 shared tensor.
static std::vector<uint8_t> create_multilayer_gguf(
    int num_layers, uint32_t K, uint32_t N, uint32_t alignment,
    std::vector<std::string>& out_tensor_names)
{
    std::vector<uint8_t> buf;
    out_tensor_names.clear();

    buf.push_back('G'); buf.push_back('G'); buf.push_back('U'); buf.push_back('F');
    append_le32(buf, 3);

    // Tensor list first (need count up front).
    std::vector<std::string> names;
    for (int l = 0; l < num_layers; l++) {
        names.push_back("model.layers." + std::to_string(l) + ".attn.q.weight");
        names.push_back("model.layers." + std::to_string(l) + ".mlp.gate.weight");
    }
    names.push_back("model.embed.weight");
    out_tensor_names = names;

    append_le64(buf, names.size());
    append_le64(buf, 2); // n_kv

    append_string(buf, "general.architecture");
    append_le32(buf, (uint32_t)GGUFValueType::STRING);
    append_string(buf, "prism");
    append_string(buf, "general.alignment");
    append_le32(buf, (uint32_t)GGUFValueType::UINT32);
    append_le32(buf, alignment);

    // Tensor info table with sequential alignment-padded offsets.
    const size_t per_block = 34;
    size_t cursor = 0;
    for (const std::string& name : names) {
        append_string(buf, name);
        append_le32(buf, 2);
        // Shared embed tensor is smaller but still 128-divisible.
        uint32_t k = (name == "model.embed.weight") ? K : K;
        uint32_t n = (name == "model.embed.weight") ? (N / 2) : N;
        append_le64(buf, k);
        append_le64(buf, n);
        append_le32(buf, 100); // Q2_0 test type id
        size_t aligned = ((cursor + alignment - 1) / alignment) * alignment;
        append_le64(buf, aligned);
        size_t this_bytes = ((size_t)k * n) / 128 * per_block;
        cursor = aligned + this_bytes;
    }

    size_t header_bytes = buf.size();
    size_t data_offset = ((header_bytes + alignment - 1) / alignment) * alignment;
    buf.insert(buf.end(), data_offset - header_bytes, 0x00);

    // Tensor data region (pad to cursor size).
    size_t data_region = ((cursor + alignment - 1) / alignment) * alignment;
    size_t data_start = buf.size();
    buf.insert(buf.end(), data_region, 0x00);

    // Fill each tensor's blocks with a layer-dependent deterministic pattern.
    cursor = 0;
    int layer_seq = 0;
    for (const std::string& name : names) {
        int idx = GGUFLayerStreamer::parse_layer_index(name);
        uint32_t n = (name == "model.embed.weight") ? (N / 2) : N;
        size_t this_bytes = ((size_t)K * n) / 128 * per_block;
        size_t aligned = ((cursor + alignment - 1) / alignment) * alignment;
        uint8_t* dst = buf.data() + data_start + aligned;
        uint8_t seed = (uint8_t)((idx < 0 ? 0xA0 : (0x10 + idx * 0x11 + layer_seq)) & 0xFF);
        for (size_t b = 0; b < this_bytes; b += per_block) {
            float scale = 0.0625f;
            __fp16 s16 = (__fp16)scale;
            uint16_t raw = 0;
            std::memcpy(&raw, &s16, sizeof(uint16_t));
            dst[b] = (uint8_t)(raw & 0xFF);
            dst[b + 1] = (uint8_t)((raw >> 8) & 0xFF);
            for (int i = 0; i < 32; i++) {
                dst[b + 2 + i] = (uint8_t)(seed ^ (uint8_t)(i * 7 + b));
            }
        }
        cursor = aligned + this_bytes;
        layer_seq++;
    }
    return buf;
}

int main() {
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << " GGUF LAYER STREAMER TEST SUITE (Task 1)" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;

    int passed = 0, total = 0;
    const uint32_t K = 256, N = 128, ALIGN = 32;
    const int NUM_LAYERS = 4;
    const char* tmp_path = "/tmp/test_multilayer_stream.gguf";

    std::vector<std::string> tensor_names;
    auto gguf_data = create_multilayer_gguf(NUM_LAYERS, K, N, ALIGN, tensor_names);

    int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    assert(fd >= 0);
    assert(write(fd, gguf_data.data(), gguf_data.size()) == (ssize_t)gguf_data.size());
    ::close(fd);

    // Test 1: layer index parsing.
    {
        total++;
        std::cout << "  RUN Layer index parsing... ";
        assert(GGUFLayerStreamer::parse_layer_index("model.layers.0.attn.q.weight") == 0);
        assert(GGUFLayerStreamer::parse_layer_index("model.layers.12.mlp.gate.weight") == 12);
        assert(GGUFLayerStreamer::parse_layer_index("model.embed.weight") == -1);
        assert(GGUFLayerStreamer::parse_layer_index("general.architecture") == -1);
        std::cout << COLOR_GREEN << "PASS" << COLOR_RESET << std::endl;
        passed++;
    }

    GGUFLayerStreamer streamer;
    {
        total++;
        std::cout << "  RUN Streamer open (resident_layer_count=1)... ";
        assert(streamer.open(tmp_path, 1));
        assert(streamer.is_open());
        assert((int)streamer.num_layers() == NUM_LAYERS);
        assert(streamer.resident_layer_count() == 1);
        std::cout << COLOR_GREEN << "PASS (" << streamer.num_layers() << " layers)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 3: bit-identical bytes vs full-resident loader path.
    {
        total++;
        std::cout << "  RUN Bit-identical streamed vs full-resident bytes... ";
        GGUFLoader full;
        assert(full.open(tmp_path));
        size_t checked = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            assert(streamer.acquire_layer(l));
            for (const std::string& name : streamer.layer_tensor_names(l)) {
                const uint8_t* s_ptr = streamer.tensor_data(name);
                const uint8_t* f_ptr = full.get_tensor_raw_data(name);
                assert(s_ptr != nullptr && f_ptr != nullptr);
                const GGUFTensorInfo* info = full.get_tensor_info(name);
                assert(info != nullptr);
                uint64_t numel = 1;
                for (uint64_t d : info->dims) numel *= d;
                size_t nbytes = (size_t)(numel / 128) * 34;
                assert(std::memcmp(s_ptr, f_ptr, nbytes) == 0);
                // Touch pages so residency accounting reflects real faults.
                volatile uint8_t sink = 0;
                for (size_t i = 0; i < nbytes; i += 4096) sink ^= s_ptr[i];
                (void)sink;
                checked++;
            }
        }
        // Shared tensor identical too.
        {
            const uint8_t* s_ptr = streamer.tensor_data("model.embed.weight");
            const uint8_t* f_ptr = full.get_tensor_raw_data("model.embed.weight");
            assert(s_ptr && f_ptr);
            const GGUFTensorInfo* info = full.get_tensor_info("model.embed.weight");
            uint64_t numel = 1;
            for (uint64_t d : info->dims) numel *= d;
            assert(std::memcmp(s_ptr, f_ptr, (size_t)(numel / 128) * 34) == 0);
        }
        full.close();
        std::cout << COLOR_GREEN << "PASS (" << checked << " tensors identical)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 4: bounded peak residency.
    {
        total++;
        std::cout << "  RUN Peak residency bounded... ";
        size_t per_layer = ((size_t)K * N) / 128 * 34 * 2; // 2 tensors/layer
        size_t total_model = streamer.total_model_bytes();
        size_t peak = streamer.peak_resident_bytes();
        std::cout << "(total=" << total_model << "B peak=" << peak
                  << "B per_layer=" << per_layer << "B) ";
        assert(total_model == per_layer * NUM_LAYERS + streamer.shared_bytes());
        assert(total_model > peak * 2); // 4 layers, only ~1 resident
        assert(peak <= per_layer + streamer.shared_bytes());
        assert(streamer.current_resident_bytes() <= per_layer + streamer.shared_bytes());
        std::cout << COLOR_GREEN << "PASS" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 5: unknown layer rejected; double-acquire is LRU-stable.
    {
        total++;
        std::cout << "  RUN Unknown-layer guard + LRU stability... ";
        assert(!streamer.acquire_layer(999));
        size_t peak_before = streamer.peak_resident_bytes();
        assert(streamer.acquire_layer(0));
        assert(streamer.acquire_layer(0));
        assert(streamer.peak_resident_bytes() == peak_before);
        std::cout << COLOR_GREEN << "PASS" << COLOR_RESET << std::endl;
        passed++;
    }

    double phys = core::memory::get_uma_phys_footprint_mb();
    std::cout << "  [telemetry] UMA phys_footprint after streaming: " << phys << " MB" << std::endl;
    std::cout << "  [telemetry] total_model=" << streamer.total_model_bytes()
              << "B peak_resident=" << streamer.peak_resident_bytes()
              << "B current_resident=" << streamer.current_resident_bytes() << "B" << std::endl;

    streamer.close();
    unlink(tmp_path);

    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    std::cout << "  Total: " << total << " Passed: " << COLOR_GREEN << passed << COLOR_RESET
              << " Failed: " << (total - passed) << std::endl;
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    return (passed == total) ? 0 : 1;
}
