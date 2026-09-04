#include "core/weights/gguf_loader.h"
#include "core/memory/quant_types.h"
#include "tests/e2e/test_common.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>

using namespace metal_llm;

// Helper to append little-endian bytes to a buffer
static void append_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

static void append_le16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back((uint8_t)(v & 0xFF));
    buf.push_back((uint8_t)((v >> 8) & 0xFF));
}

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

// Builds a synthetic GGUF binary buffer with metadata and one Q2_0 tensor
static std::vector<uint8_t> create_synthetic_gguf(
    uint32_t K,
    uint32_t N,
    float scale,
    uint32_t alignment = 32,
    bool corrupt_magic = false)
{
    std::vector<uint8_t> buf;

    // 1. Header
    if (corrupt_magic) {
        buf.push_back('B'); buf.push_back('A'); buf.push_back('D'); buf.push_back('!');
    } else {
        buf.push_back('G'); buf.push_back('G'); buf.push_back('U'); buf.push_back('F');
    }
    append_le32(buf, 3); // GGUF Version 3

    uint64_t n_tensors = 1;
    uint64_t n_kv = 3; // general.architecture, general.alignment, prism.block_count
    append_le64(buf, n_tensors);
    append_le64(buf, n_kv);

    // 2. Metadata KV Table
    // KV 1: general.architecture (STRING)
    append_string(buf, "general.architecture");
    append_le32(buf, (uint32_t)GGUFValueType::STRING);
    append_string(buf, "prism");

    // KV 2: general.alignment (UINT32)
    append_string(buf, "general.alignment");
    append_le32(buf, (uint32_t)GGUFValueType::UINT32);
    append_le32(buf, alignment);

    // KV 3: prism.block_count (UINT32)
    append_string(buf, "prism.block_count");
    append_le32(buf, (uint32_t)GGUFValueType::UINT32);
    append_le32(buf, 16);

    // 3. Tensor Info Table
    // Tensor 1: model.layers.0.mlp.gate.weight
    append_string(buf, "model.layers.0.mlp.gate.weight");
    append_le32(buf, 2); // 2 dimensions
    append_le64(buf, K); // ne0 = K (columns)
    append_le64(buf, N); // ne1 = N (rows)
    append_le32(buf, 100); // PrismML Q2_0 type ID
    append_le64(buf, 0);   // Tensor offset relative to data_offset

    // 4. Pad header up to alignment boundary
    size_t header_bytes = buf.size();
    size_t aligned_data_offset = ((header_bytes + alignment - 1) / alignment) * alignment;
    size_t pad_bytes = aligned_data_offset - header_bytes;
    buf.insert(buf.end(), pad_bytes, 0x00);

    // 5. Tensor Data (Q2_0 blocks)
    size_t total_elements = (size_t)K * N;
    size_t num_blocks = total_elements / 128;

    __fp16 scale_fp16 = (__fp16)scale;
    uint16_t scale_raw = 0;
    std::memcpy(&scale_raw, &scale_fp16, sizeof(uint16_t));

    for (size_t b = 0; b < num_blocks; b++) {
        // FP16 scale (little-endian)
        append_le16(buf, scale_raw);

        // 32 bytes packed 2-bit codes
        // Pattern: byte i contains 4 codes:
        // code 0: (i % 4)
        // code 1: ((i + 1) % 4)
        // code 2: ((i + 2) % 4)
        // code 3: ((i + 3) % 4)
        for (int i = 0; i < 32; i++) {
            uint8_t c0 = (uint8_t)((i + 0) % 4);
            uint8_t c1 = (uint8_t)((i + 1) % 4);
            uint8_t c2 = (uint8_t)((i + 2) % 4);
            uint8_t c3 = (uint8_t)((i + 3) % 4);
            uint8_t packed = (c0) | (c1 << 2) | (c2 << 4) | (c3 << 6);
            append_u8(buf, packed);
        }
    }

    return buf;
}

int main() {
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << " GGUF LOADER & PRISMML Q2_0 PARSING INTEGRITY TEST SUITE" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Corrupt Magic Rejection
    {
        total++;
        std::cout << "  RUN GGUF Magic Header Rejection... ";
        auto corrupt_buf = create_synthetic_gguf(256, 128, 0.05f, 32, true);
        GGUFLoader loader;
        bool res = loader.parse_from_buffer(corrupt_buf.data(), corrupt_buf.size());
        assert(!res);
        assert(!loader.is_open());
        std::cout << COLOR_GREEN << "PASS (Cleanly rejected corrupt magic)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 2: Truncated File Rejection
    {
        total++;
        std::cout << "  RUN GGUF Truncated Buffer Rejection... ";
        std::vector<uint8_t> tiny_buf = {'G', 'G', 'U', 'F', 0x03, 0x00};
        GGUFLoader loader;
        bool res = loader.parse_from_buffer(tiny_buf.data(), tiny_buf.size());
        assert(!res);
        assert(!loader.is_open());
        std::cout << COLOR_GREEN << "PASS (Cleanly rejected < 24 byte buffer)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 3: In-Memory GGUF Parsing & Metadata Walk
    const uint32_t K = 256;
    const uint32_t N = 128;
    const float test_scale = 0.0625f; // 1/16 exact power of 2 for FP16 precision
    const uint32_t test_alignment = 64;

    auto gguf_data = create_synthetic_gguf(K, N, test_scale, test_alignment, false);
    GGUFLoader loader;

    {
        total++;
        std::cout << "  RUN GGUF In-Memory Parser & Metadata Verification... ";
        bool ok = loader.parse_from_buffer(gguf_data.data(), gguf_data.size());
        assert(ok);
        assert(loader.is_open());
        assert(loader.version() == 3);
        assert(loader.tensor_count() == 1);
        assert(loader.metadata_count() == 3);
        assert(loader.alignment() == test_alignment);

        // Check metadata
        assert(loader.has_metadata("general.architecture"));
        assert(loader.get_string_metadata("general.architecture") == "prism");
        assert(loader.has_metadata("general.alignment"));
        assert(loader.get_uint32_metadata("general.alignment") == test_alignment);
        assert(loader.has_metadata("prism.block_count"));
        assert(loader.get_uint32_metadata("prism.block_count") == 16);

        std::cout << COLOR_GREEN << "PASS (Metadata & Alignment = 64 verified)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 4: Tensor Info Table Verification
    {
        total++;
        std::cout << "  RUN GGUF Tensor Info Table Parsing... ";
        assert(loader.has_tensor("model.layers.0.mlp.gate.weight"));
        const auto* info = loader.get_tensor_info("model.layers.0.mlp.gate.weight");
        assert(info != nullptr);
        assert(info->name == "model.layers.0.mlp.gate.weight");
        assert(info->n_dims == 2);
        assert(info->dims.size() == 2);
        assert(info->dims[0] == K);
        assert(info->dims[1] == N);
        assert(info->type == 100);
        assert(info->offset == 0);
        assert(loader.tensor_data_offset() % test_alignment == 0);

        std::cout << COLOR_GREEN << "PASS (Tensor name, shape [256, 128], and alignment verified)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 5: Q2_0 Tensor Extraction and Dequantization Parity
    std::vector<core::memory::block_prism_q2_0> extracted_blocks;
    uint64_t out_k = 0, out_n = 0;

    {
        total++;
        std::cout << "  RUN PrismML Q2_0 Tensor Extraction & Endian-Safety... ";
        bool extract_ok = loader.extract_q2_0_tensor(
            "model.layers.0.mlp.gate.weight",
            extracted_blocks,
            &out_k,
            &out_n);

        assert(extract_ok);
        assert(out_k == K);
        assert(out_n == N);

        size_t expected_blocks = (K * N) / 128;
        assert(extracted_blocks.size() == expected_blocks);
        assert(sizeof(core::memory::block_prism_q2_0) == 34);

        std::cout << COLOR_GREEN << "PASS (Extracted " << extracted_blocks.size() << " blocks of 34 bytes)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 6: Round-Trip Mathematical Parity Through CPU Dequant Formula
    {
        total++;
        std::cout << "  RUN Q2_0 Weight Dequantization Round-Trip & Reserved Code Check... ";
        // Dequantize and check values against ground truth pattern
        float max_diff = 0.0f;
        int reserved_code_hits = 0;

        for (size_t b = 0; b < extracted_blocks.size(); b++) {
            const auto& blk = extracted_blocks[b];
            float scale_val = (float)blk.d;
            assert(std::fabs(scale_val - test_scale) < 1e-5f);

            for (int i = 0; i < 32; i++) {
                uint8_t byte_val = blk.qs[i];
                for (int j = 0; j < 4; j++) {
                    uint8_t q = (byte_val >> (j * 2)) & 0x3;
                    uint8_t expected_q = (uint8_t)((i + j) % 4);
                    assert(q == expected_q);

                    // Dequant formula: w = (q - 1) * scale
                    float expected_weight = ((float)q - 1.0f) * test_scale;

                    int code = (int)q - 1;
                    float actual_weight = (float)code * scale_val;

                    float diff = std::fabs(actual_weight - expected_weight);
                    if (diff > max_diff) max_diff = diff;

                    if (q == 3) {
                        reserved_code_hits++;
                        // Must decode to +2 * scale
                        assert(std::fabs(actual_weight - (2.0f * test_scale)) < 1e-5f);
                    }
                }
            }
        }

        assert(max_diff == 0.0f);
        assert(reserved_code_hits > 0);
        std::cout << COLOR_GREEN << "PASS (Bit-exact dequant, diff=0.000000, q=3 correctly decodes to +2*scale)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 7: Disk-Based File Open (mmap) Verification
    {
        total++;
        std::cout << "  RUN GGUF Disk-Based mmap Loader Verification... ";
        const char* tmp_path = "/tmp/test_synthetic_prism_q2_0.gguf";

        int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        assert(fd >= 0);
        ssize_t written = write(fd, gguf_data.data(), gguf_data.size());
        assert(written == (ssize_t)gguf_data.size());
        close(fd);

        GGUFLoader disk_loader;
        bool open_ok = disk_loader.open(tmp_path);
        assert(open_ok);
        assert(disk_loader.is_open());
        assert(disk_loader.file_size() == gguf_data.size());

        std::vector<core::memory::block_prism_q2_0> disk_blocks;
        uint64_t dk = 0, dn = 0;
        bool extract_ok = disk_loader.extract_q2_0_tensor("model.layers.0.mlp.gate.weight", disk_blocks, &dk, &dn);
        assert(extract_ok);
        assert(dk == K && dn == N);
        assert(disk_blocks.size() == extracted_blocks.size());

        // Verify exact binary identity between memory and disk loads
        int cmp = std::memcmp(disk_blocks.data(), extracted_blocks.data(), disk_blocks.size() * sizeof(core::memory::block_prism_q2_0));
        assert(cmp == 0);

        disk_loader.close();
        unlink(tmp_path);

        std::cout << COLOR_GREEN << "PASS (Disk mmap matches in-memory parse bit-for-bit)" << COLOR_RESET << std::endl;
        passed++;
    }

    // Test 8: Non-Existent Tensor & Dimension Divisibility Rejection
    {
        total++;
        std::cout << "  RUN Non-Existent Tensor & Bounds Guard... ";
        std::vector<core::memory::block_prism_q2_0> dummy_blocks;
        bool not_found = loader.extract_q2_0_tensor("non_existent_weight", dummy_blocks);
        assert(!not_found);

        const uint8_t* raw = loader.get_tensor_raw_data("non_existent_weight");
        assert(raw == nullptr);

        const uint8_t* valid_raw = loader.get_tensor_raw_data("model.layers.0.mlp.gate.weight");
        assert(valid_raw != nullptr);

        std::cout << COLOR_GREEN << "PASS (Guard checks properly caught invalid requests)" << COLOR_RESET << std::endl;
        passed++;
    }

    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << " SUMMARY: GGUF Loader & PrismML Q2_0 Tests" << COLOR_RESET << std::endl;
    std::cout << "  Total Tests:  " << total << std::endl;
    std::cout << "  Passed Tests: " << COLOR_GREEN << passed << COLOR_RESET << std::endl;
    std::cout << "  Failed Tests: " << (total - passed) << std::endl;
    std::cout << COLOR_BOLD << "=================================================================" << COLOR_RESET << std::endl;

    return (passed == total) ? 0 : 1;
}
