# m4-prefill-engine
**A complete inference architecture for Apple Silicon that unifies prefill, decode, and out-of-core streaming at MLX-native speeds.**

> **Live Documentation:** [mohamedhossammohamed.github.io/m4-prefill-engine](https://mohamedhossammohamed.github.io/m4-prefill-engine/)
> **Hardware Target:** Apple M4 MacBook Air (10-core GPU, 16GB Unified Memory)
> **Version:** v0.2

---

## The Problem: The Apple Silicon Inference Tradeoff

Local LLM inference on Apple Silicon currently forces a painful choice between three competing constraints:

1. **For speed:** Use Apple's MLX framework, but you are locked into MLX's proprietary 4-bit format and miss the massive, diverse GGUF ecosystem.
2. **For format flexibility:** Use `llama.cpp` for GGUF compatibility, but accept a 20–40% prefill speed penalty on Apple Silicon due to generic tensor routing.
3. **For long contexts:** Either crash with Out-Of-Memory (OOM) errors when contexts exceed physical RAM, or accept severe performance degradation from OS-level SSD swap.

This project asks: **Can a single, low-level Metal engine eliminate all three tradeoffs simultaneously?**

---

## The Solution: A Unified Inference Architecture (v0.2)

By bypassing high-level framework abstractions and writing custom Metal shaders directly for the M4's Load-Store Units (LSU) and Hardware Matrix Coprocessor, this engine introduces three architectural pillars that solve the tradeoff.

### Pillar 1: Compute-Bound Prefill (The 4-Brick Architecture)
By unlocking Apple's hidden Hardware Matrix Coprocessor (`simdgroup_matrix`) and saturating the LSU with 128-bit vector firehoses, prefill becomes **compute-bound** rather than memory-bound. This frees resources for concurrent decode operations.

*   **Brick 1 (Hardware MMA):** Transitioned from standard Vector ALUs (~7.4 TFLOPS) to the 16.8 TFLOPS Hardware Matrix Coprocessor using 8×8 `simdgroup_matrix` fragments.
*   **Brick 2 (Memory Ingestion):** 2D block-swizzled DRAM layout with 128-bit cooperative firehoses and padded threadgroup SRAM (`[64][36]`) to guarantee 1-cycle conflict-free bank broadcasts.
*   **Brick 3 (Dual-SIMD SwiGLU):** Split Gate and Up projections across separate SIMDgroups to share input activations in SRAM, eliminating 7.68 GB of DRAM churn per 8B model layer.
*   **Brick 4 (Barrier-Free FlashAttention):** Replaced expensive `threadgroup_barrier()` calls with `simd_shuffle_down` register butterfly trees for online softmax reductions, paired with dynamic Q8_0 KV cache compression.

### Pillar 2: Universal Quantization Router
A modular router decodes six distinct quantization formats on-the-fly, feeding them into the same hardware-saturated pipeline. This achieves MLX-native speeds across the broader open-source ecosystem.

| Format | Bits/Weight | Status | Performance vs MLX |
| :--- | :---: | :--- | :--- |
| **Q4_0** (GGUF) | 4.5 | Production | Parity with MLX |
| **MLX 4-bit** | 5.0 | Production | **1.05x–1.25x faster** on unaligned boundaries |
| **Q4_K** (GGUF Super-Blocks) | 4.5 | Production | Parity with MLX |
| **Variable-Rate Affine** | 3–5 | Production | Parity with MLX |
| **EXL3** (Hierarchical Codebook) | 4.5 | Production | Parity with MLX |
| **Ternary 1.58-bit** (BitNet) | 3.0 (1.58 entropy) | Production | **1.3x–2.3x faster** at mid-lengths (SLC fit) |

*Note on Ternary 1.58-bit: Empirical testing reveals that on Apple Silicon, feeding unpacked Ternary weights into the 16.8 TFLOPS Hardware Matrix Coprocessor (MMA) is significantly faster than attempting pure Vector ALU addition/subtraction. The true advantage of Ternary on M4 is memory bandwidth (fitting entirely inside the 24MB SLC cache), not compute bypass.*

### Pillar 3: 1M-Token Out-of-Core SSD Streaming
When contexts exceed physical RAM (16GB), the engine treats the NVMe SSD as an extension of Unified Memory.

*   **True NVMe DMA:** Utilizes `F_NOCACHE` with strictly 16KB page-aligned (`posix_memalign`) buffers to bypass the macOS Unified Buffer Cache (UBC), achieving 2.0–3.0 GB/s physical read throughput.
*   **Chunked FlashAttention:** Online softmax running statistics are persisted to global memory between SSD chunks, enabling mathematically exact attention across arbitrarily long contexts.
*   **Dual 128MB Ring Buffer:** Overlaps GPU compute with SSD streaming, hiding storage latency behind the Matrix Coprocessor.

---

## Honest Benchmarking Methodology

All benchmarks adhere to strict systems-engineering rigor to ensure physical reality matches published claims:

*   **Cold-Cache Isolation:** Mandatory 32MB SLC flushes before every format test to prevent cache pollution.
*   **True GPU Timestamps:** Latencies measured using Metal's native `GPUStartTime` / `GPUEndTime`, isolating pure kernel compute from CPU driver overhead.
*   **Edge-Case Validation:** Non-aligned sequence lengths (M ∈ [33, 127, 128, 129, 512, 1023, 1024, 2047, 2048]) verify causal masking and boundary guards.
*   **Honest Verification:** CPU double-precision gold standards are strictly executed for M ≤ 2048. Larger scales (up to 1M) are explicitly labeled as GPU-only streaming, as O(M²) CPU verification is physically infeasible at that scale.

### 1M-Token SSD Streaming Telemetry (Apple M4, 16GB UMA)

| Context (M) | Execution Mode | End-to-End | GPU Compute | SSD Read BW | Peak UMA Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **64K** | Mode A (Full Causal) | 9.48 s | 9.23 s | 2.0 GB/s | 7.0 GB |
| **128K** | Mode A (Full Causal) | 37.09 s | 36.67 s | 2.4 GB/s | 11.1 GB |
| **1M** | Mode B (Spec K=64) | **1.72 s** | **1.67 s** | **2.7 GB/s** | **12.5 GB** |

*Note: Full causal streaming (Mode A) is capped at 128K tokens due to state buffer constraints. 256K to 1M contexts utilize speculative burst verification (Mode B). At 1M tokens, the engine consumes 12.5 GB of physical UMA (`phys_footprint`), leaving ~3.5 GB for macOS.*

---

## ⚠️ Target Audience & Usage Disclaimer

**This is a research artifact and proof-of-concept.**

It is **not** intended as a drop-in replacement for everyday `llama.cpp` users, Ollama, or consumer LLM frontends. It currently requires:
*   Manual compilation with the Apple Metal toolchain (`make`).
*   Command-line execution.
*   Synthetic weight generation (matching LLaMA shapes) to isolate pure silicon execution from disk I/O.

The goal of this repository is to provide a verified, open-source baseline for the community. These optimizations are intended to be upstreamed into mainstream frameworks (`ggml-metal`, `MLX`) or adopted as specialized hardware configurations.

---

## Building and Running

### Prerequisites
*   Apple Silicon Mac (M4/M5) running macOS 14.0+
*   Xcode Command Line Tools (`xcode-select --install`)

### Compilation & Execution

```bash
git clone https://github.com/mohamedhossammohamed/m4-prefill-engine.git
cd m4-prefill-engine
make clean && make

# 1. Hardware calibration & baseline
./bench_m4

# 2. Queue-saturated double-buffered GEMM
./pipelined_bench

# 3. FlashAttention (FP16 vs Q8_0 KV Cache)
./flash_attn_bench

# 4. Full end-to-end 1B prefill layer
./unified_prefill_engine

# 5. Universal Quantization Router (6 formats)
./bench_universal_router

# 6. 1M-token SSD streaming engine
./bench_streaming_1m

# 7. 60-second thermal stress test
./thermal_stress_test
```

---

## Licensing

### The Official License
Copyright 2026 Mohammed Hossam.  
This project is officially and legally licensed under the **Apache License 2.0**. You are free to use, modify, and distribute this code in accordance with the terms of the Apache 2.0 license.

### Officially the Unofficial License of the Project
In addition to the Apache 2.0 license, this project proudly operates under the [`no-theo-license`](https://github.com/maria-rcks/no-theo-license) until **March 31, 2027**.

Under the strict legal statutes of this unofficial license, the software is open to the entire world, corporations, and alien civilizations—with the sole exception of Theo. **Theo is strictly forbidden from compiling, executing, reading, or thinking about this repository until his next birthday on March 31, 2027.** Once the clock strikes midnight on that date, the restriction shall be lifted.

---

## A Note on Citations & Future Use

I am currently early in my engineering career, and building this engine has been a massive learning experience.

If the ideas, techniques, or specific hardware-level optimizations from this repository (such as the M4 LSU saturation methods, Universal Quantization Router, 1M SSD streaming architecture, or Metal-specific prefill routing) are adapted, ported to other silicon architectures (AMD/Nvidia/Intel), or used to improve decoding phases in other software, I humbly ask for a **visible citation, link, or mention** in your project's documentation, blog post, or research paper.

Any visibility that helps a junior engineer grow and find their footing in the systems engineering community is deeply and genuinely appreciated. Thank you for reading, testing, and building.

---

## Contact & Discussion

If you want to discuss Metal optimization, Apple Silicon memory hierarchies, or LLM inference, feel free to reach out:

*   **X (Twitter):** [Mohammed Hossam (@MohamedHz72007)](https://x.com/MohamedHz72007)
*   **GitHub:** [mohamedhossammohamed](https://github.com/mohamedhossammohamed)
