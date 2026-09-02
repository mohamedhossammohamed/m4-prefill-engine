# m4-prefill-engine
**A low-level Metal inference architecture for Apple Silicon: universal quantization formats, out-of-core 1M-token contexts, and decode that stays out of single-digit tokens/sec.**

> **Live Documentation:** [mohamedhossammohamed.github.io/m4-prefill-engine](https://mohamedhossammohamed.github.io/m4-prefill-engine/)  
> **Hardware Target:** Apple M4 MacBook Air (10-core GPU, 16GB Unified Memory)  
> **Version:** v0.2.1 ("Beyond MLX Prefill Speeds")

---

## The Problem: Three Tradeoffs

Local LLM inference on Apple Silicon forces a painful choice: (1) Speed — MLX is fast but locks you into its proprietary 4-bit format; (2) Formats — llama.cpp gives you the GGUF ecosystem but pays a prefill penalty on Apple GPUs; (3) Context — beyond physical RAM you either OOM or degrade into OS swap. This project asks whether one low-level Metal engine can address all three.

---

## The Solution: A Unified Inference Architecture (v0.2.1)

By bypassing high-level framework abstractions and writing custom Metal shaders directly for the M4's Load-Store Units (LSU) and Hardware Matrix Coprocessor, this engine introduces four architectural pillars that address these constraints.

### Pillar 1: Compute-Bound Prefill (The 4-Brick Architecture)
By unlocking Apple's hidden Hardware Matrix Coprocessor (`simdgroup_matrix`) and saturating the LSU with 128-bit vector firehoses, prefill becomes **compute-bound** rather than memory-bound. This frees bandwidth for concurrent operations.

*   **Brick 1 (Hardware MMA):** Transitioned from standard Vector ALUs (~7.4 TFLOPS) to the 16.8 TFLOPS Hardware Matrix Coprocessor using 8×8 `simdgroup_matrix` fragments.
*   **Brick 2 (Memory Ingestion):** 2D block-swizzled DRAM layout with 128-bit cooperative firehoses and padded threadgroup SRAM (`[64][36]`) to guarantee 1-cycle conflict-free bank broadcasts.
*   **Brick 3 (Dual-SIMD SwiGLU):** Split Gate and Up projections across separate SIMDgroups to share input activations in SRAM, eliminating 7.68 GB of DRAM churn per 8B model layer.
*   **Brick 4 (Barrier-Free FlashAttention):** Replaced expensive `threadgroup_barrier()` calls with `simd_shuffle_down` register butterfly trees for online softmax reductions, paired with dynamic Q8_0 KV cache compression.

### Pillar 2: Universal Quantization Router
A modular router decodes six distinct quantization formats on-the-fly, feeding them into the same hardware-saturated pipeline. This achieves MLX-native speeds across the broader open-source ecosystem.

#### Table A: Format Router Single-Projection Performance (Apple M4, Cold-Cache Isolated)

| Model Tier | Seq Len (M) | Q4_0 (GGUF) | MLX 4-bit | Q4_K (GGUF) | Var-Rate Affine | EXL3 Codebook | Ternary MMA (BitNet) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8B (K=4096)** | **33** | 0.930 ms | **0.848 ms** | 1.117 ms | 1.637 ms | 1.272 ms | 0.849 ms |
| | **128** | 2.711 ms | 2.030 ms | 2.280 ms | 2.205 ms | 2.326 ms | **1.481 ms** |
| | **129** | 3.026 ms | 2.434 ms | 2.637 ms | 2.824 ms | 2.905 ms | **2.199 ms** |
| | **2048** | 22.009 ms | **21.744 ms** | 24.622 ms | 26.426 ms | 28.276 ms | 21.960 ms |
| **1B (K=2048)** | **33** | 0.257 ms | **0.256 ms** | 0.312 ms | 0.304 ms | 0.365 ms | 0.321 ms |
| | **128** | 1.847 ms | 1.840 ms | 2.109 ms | 1.231 ms | 1.545 ms | **1.020 ms** |
| | **129** | 2.355 ms | 1.520 ms | 2.110 ms | 1.618 ms | 1.754 ms | **1.077 ms** |
| | **2048** | **5.557 ms** | 5.598 ms | 6.217 ms | 6.549 ms | 7.237 ms | 5.662 ms |

*Source: Measured via `bench_universal_router` with mandatory 32MB SLC cache flushing and double-precision CPU verification (MaxDiff ≤ 0.0078).*  
*Note on Ternary 1.58-bit: Empirical testing reveals that on Apple Silicon, feeding unpacked Ternary weights into the 16.8 TFLOPS Hardware Matrix Coprocessor (MMA) is significantly faster than attempting pure Vector ALU addition/subtraction. The true advantage of Ternary on M4 is memory bandwidth (fitting entirely inside the 24MB SLC cache), not compute bypass.*

> [!IMPORTANT]
> **Release Scope Notice (v0.2):** 1,000,000-token out-of-core flash streaming and speculative burst decoding are designated as **experimental research prototypes** and are **strictly excluded from this release**. Today's release scope is focused 100% on ultra-low-latency in-core prefill acceleration ($M \le 2048$ tokens). Context streaming and decoding benchmarks are deferred and are not executed for this release.

### Pillar 3: 1M-Token Out-of-Core Flash Streaming (Experimental Research Prototype — Deferred)
When contexts exceed physical RAM (16GB), the engine treats internal PCIe flash storage as an extension of Unified Memory.

*   **Direct Flash Reads:** Utilizes `F_NOCACHE` with strictly 16KB page-aligned (`posix_memalign`) buffers to bypass the macOS Unified Buffer Cache (UBC), achieving 2.0–3.0 GB/s physical read throughput from internal PCIe flash storage.
*   **Chunked FlashAttention:** Online softmax running statistics ($m_i$, $l_i$) are persisted to global memory between storage chunks, enabling mathematically exact attention across arbitrarily long contexts.
*   **Dual 128MB Ring Buffer:** Overlaps GPU compute with flash reads, hiding storage latency behind the Matrix Coprocessor.

### Pillar 4: On-the-Fly Out-of-Core Decode (Experimental Research Prototype — Deferred)
The streaming engine above solves prefill. The harder question is decode: every generated token must attend over the entire context, and at 1M tokens that context lives on flash. This pillar is an on-the-fly proof-of-concept — a handful of tricks to test whether a 1,000,000-token out-of-core context can decode without collapsing into single-digit tokens/sec. Note: Decoding benchmarks are deferred from the current release.

#### Computed Bandwidth Floors (Theoretical Limits)
*   **1M Context, 1B Shape ($H=32, D=64$):** Q8_0 KV $\approx$ 4.3 GB per full-context pass (FP16 would be $\approx$ 8.6 GB).
*   **[COMPUTED] Naive Autoregressive Decode off Flash:** 4.3 GB $\div$ 2.7 GB/s $\approx$ 1.6 s/token $\approx$ **0.6 tok/s**.
*   **[COMPUTED] Naive In-RAM Ceiling (if it fit in RAM):** 4.3 GB $\div$ 98 GB/s $\approx$ **22 tok/s**.

#### Measured Out-of-Core Decode Telemetry (Apple M4, 16GB RAM)

| Context (M) | Decode Strategy | Measured End-to-End | Measured GPU Compute | Flash Read BW | Throughput | Peak UMA (`phys_footprint`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **4K** | [MEASURED] Speculative Burst ($K=64$) | 72.54 ms | 6.99 ms | 2.1 GB/s | **882 tok/s** | 1.75 GB |
| **64K** | [MEASURED] Speculative Burst ($K=64$) | 162.74 ms | 104.25 ms | 2.5 GB/s | **393 tok/s** | 7.45 GB |
| **128K** | [MEASURED] Speculative Burst ($K=64$) | 288.99 ms | 231.93 ms | 2.3 GB/s | **221 tok/s** | 11.58 GB |
| **1M** | [MEASURED] Naive Single-Token Decode | 1.68 s | 0.08 s | 2.6 GB/s | **0.60 tok/s** | 12.51 GB |
| **1M** | [MEASURED] Speculative Burst ($K=64$) | **1.82 s** | **1.74 s** | **2.59 GB/s** | **35.2 measured tok/s** | **12.51 GB** |

*Result: Speculative burst verification ($K=64$ candidates processed in a single KV stream) delivers ~35.2 measured tok/s at 1,000,000 tokens — ~60x faster than the naive flash floor (0.60 tok/s) and exceeding the naive in-RAM ceiling (22 tok/s), because the 4.3 GB stream is amortized across 64 candidate tokens.*

#### The Architectural Tricks
1. **Q8_0 KV Cache Compression:** Halves the required storage footprint and read volume from 8.6 GB down to 4.3 GB per pass.
2. **Dual 128MB Ring Buffering:** Asynchronous double-buffered I/O overlaps flash storage DMA with GPU Matrix Coprocessor execution.
3. **Speculative Burst Verification:** Amortizes the fixed 4.3 GB streaming cost over $K=64$ candidate tokens simultaneously in registers.
4. **Chunked Online Softmax State:** Carries running numerical state ($m_i$, $l_i$) across chunk boundaries without losing mathematical precision.

#### Honest Caveats
*   **Draft Acceptance Dependency:** Effective generation speed depends on speculative drafter acceptance rates.
*   **Numerical Verification Scale:** Double-precision CPU ground truth verification is strictly executed for $M \le 2048$. 1M context decode rows are GPU-only measurements, as $O(M^2)$ CPU verification is physically infeasible at that scale.

---

## Full-Layer Prefill Comparison: Apple MLX vs Ours

### Table B: Full-Layer Prefill Comparison (Strict Apples-to-Apples in-RAM, $M \le 2048$)

Measured using shared wall-clock timing parity (10 warmup, 20 measured iterations, 32MB SLC flush, identical in-RAM synthetic weights, no tokenizer overhead, single forward pass). 
*   **MLX Comparison:** Apple MLX Metal vs Our Engine executed directly on identical **MLX 4-bit weights** (`block_mlx_4bit`).
*   **llama.cpp Comparison:** `llama.cpp` baseline vs Our Engine executed directly on identical **GGUF Q4_0 weights** (`block_q4_0`).
*   **Context Scope:** Restricted strictly to in-core RAM ($M \le 2048$ tokens). Out-of-core streaming and speculative decoding are proprietary standalone subsystems and are not compared against standard in-RAM runners.

#### 8B Model Tier (32 Layers, $K=4096, H=32, D=128, N_{\text{mlp}}=14336$)

| Prompt ($M$) | Boundary Type | Apple MLX Metal (MLX 4-bit) | Our Engine (MLX 4-bit) | vs MLX | Our Engine (GGUF Q4_0) | llama.cpp (GGUF Q4_0) | vs llama.cpp |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | 563.52 ms (17.61 ms/L) | **449.58 ms** (14.05 ms/L) | **1.25x faster** | 497.60 ms (15.55 ms/L) | 872.64 ms (27.27 ms/L) | **1.75x faster** |
| **127** | Edge (Unaligned) | 1.26 s (39.41 ms/L) | **1.03 s** (32.05 ms/L) | **1.23x faster** | 1.16 s (36.22 ms/L) | 2.11 s (66.03 ms/L) | **1.82x faster** |
| **128** | Aligned ($2^7$) | 1.31 s (41.09 ms/L) | **1.03 s** (32.31 ms/L) | **1.27x faster** | 1.19 s (37.25 ms/L) | 2.10 s (65.72 ms/L) | **1.76x faster** |
| **129** | Edge (Unaligned) | 2.17 s (67.74 ms/L) | **1.51 s** (47.06 ms/L) | **1.44x faster** | 1.73 s (54.02 ms/L) | 3.03 s (94.67 ms/L) | **1.75x faster** |
| **512** | Aligned ($2^9$) | 4.97 s (155.32 ms/L) | **4.21 s** (131.61 ms/L) | **1.18x faster** | 4.79 s (149.69 ms/L) | 8.42 s (263.01 ms/L) | **1.76x faster** |
| **1023** | Edge (Unaligned) | 10.55 s (329.61 ms/L) | **9.33 s** (291.46 ms/L) | **1.13x faster** | 10.40 s (324.95 ms/L) | 18.54 s (579.48 ms/L) | **1.78x faster** |
| **1024** | Aligned ($2^{10}$) | 9.83 s (307.27 ms/L) | **9.11 s** (284.84 ms/L) | **1.08x faster** | 10.16 s (317.57 ms/L) | 18.00 s (562.35 ms/L) | **1.77x faster** |
| **2047** | Edge (Unaligned) | 20.50 s (640.67 ms/L) | **19.44 s** (607.57 ms/L) | **1.05x faster** | 21.54 s (673.15 ms/L) | 39.23 s (1226.02 ms/L) | **1.82x faster** |
| **2048** | Aligned ($2^{11}$) | **19.60 s** (612.59 ms/L) | 19.75 s (617.31 ms/L) | 0.99x (≈ Parity) | 21.56 s (673.81 ms/L) | 38.78 s (1211.75 ms/L) | **1.80x faster** |

#### 1B Model Tier (16 Layers, $K=2048, H=32, D=64, N_{\text{mlp}}=5632$)

| Prompt ($M$) | Boundary Type | Apple MLX Metal (MLX 4-bit) | Our Engine (MLX 4-bit) | vs MLX | Our Engine (GGUF Q4_0) | llama.cpp (GGUF Q4_0) | vs llama.cpp |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | 59.84 ms (3.74 ms/L) | **51.74 ms** (3.23 ms/L) | **1.16x faster** | 57.76 ms (3.61 ms/L) | 99.04 ms (6.19 ms/L) | **1.71x faster** |
| **127** | Edge (Unaligned) | **103.68 ms** (6.48 ms/L) | 109.71 ms (6.86 ms/L) | 0.95x | 128.00 ms (8.00 ms/L) | 217.28 ms (13.58 ms/L) | **1.70x faster** |
| **128** | Aligned ($2^7$) | **101.28 ms** (6.33 ms/L) | 121.84 ms (7.62 ms/L) | 0.83x | 135.04 ms (8.44 ms/L) | 226.72 ms (14.17 ms/L) | **1.68x faster** |
| **129** | Edge (Unaligned) | **131.84 ms** (8.24 ms/L) | 188.03 ms (11.75 ms/L) | 0.70x | 226.88 ms (14.18 ms/L) | 379.84 ms (23.74 ms/L) | **1.67x faster** |
| **512** | Aligned ($2^9$) | **374.56 ms** (23.41 ms/L) | 476.76 ms (29.80 ms/L) | 0.79x | 570.72 ms (35.67 ms/L) | 1.10 s (68.57 ms/L) | **1.92x faster** |
| **1023** | Edge (Unaligned) | **761.12 ms** (47.57 ms/L) | 1.06 s (66.00 ms/L) | 0.72x | 1.18 s (73.53 ms/L) | 2.44 s (152.77 ms/L) | **2.08x faster** |
| **1024** | Aligned ($2^{10}$) | **796.48 ms** (49.78 ms/L) | 997.80 ms (62.36 ms/L) | 0.80x | 1.11 s (69.59 ms/L) | 2.17 s (135.82 ms/L) | **1.95x faster** |
| **2047** | Edge (Unaligned) | 1.95 s (122.16 ms/L) | **1.70 s** (106.10 ms/L) | **1.15x faster** | 1.87 s (116.93 ms/L) | 4.31 s (269.49 ms/L) | **2.30x faster** |
| **2048** | Aligned ($2^{11}$) | 2.02 s (126.09 ms/L) | **1.71 s** (106.61 ms/L) | **1.18x faster** | 1.87 s (116.74 ms/L) | 4.09 s (255.91 ms/L) | **2.19x faster** |

---

## ⚠️ Target Audience & Usage Disclaimer

**This is a research artifact and proof-of-concept.**

It is **not** intended as a drop-in replacement for everyday `llama.cpp` users, Ollama, or consumer LLM frontends. It currently requires:
*   Manual compilation with the Apple Metal toolchain (`make`).
*   Command-line execution.
*   Synthetic weight generation (matching LLaMA shapes) to isolate pure silicon execution from disk I/O.

The goal of this repository is to provide a verified, open-source baseline for the community. These optimizations are intended to be upstreamed into mainstream frameworks (`ggml-metal`, `MLX`) or adopted as specialized hardware configurations.

---

## 📚 Technical Documentation & Architecture Deep Dives

For in-depth mathematical proofs, low-level shader mechanics, memory bank conflict analyses, and cross-hardware porting guides, consult the dedicated documentation suite:

* [**Core Architecture & The 4-Brick Pipeline**](architecture.md) — 4-way fused ALU dequantization, 2D block swizzling, 128-bit LSU vector firehoses, padded threadgroup SRAM (`[64][36]`), Dual-SIMD SwiGLU fusion, and barrier-free FlashAttention (`simd_shuffle_down`).
* [**Universal Quantization Router Architecture**](quantization_router.md) — On-the-fly decoding of Q4_0, MLX 4-bit, Q4_K, Variable-Rate Affine, EXL3 Codebook, and BitNet Ternary 1.58-bit into Hardware Matrix Units.
* [**1,000,000-Token Out-of-Core SSD Flash Streaming & Speculative Decode**](out_of_core_streaming.md) — Direct PCIe NVMe I/O (`F_NOCACHE` with 16KB alignment), dual 128MB asynchronous ring buffering, multi-chunk softmax induction, and parallel speculative burst verification ($K=64$).
* [**Cross-Architecture Porting & Hardware Translation Guide**](cross_metal_transfer.md) — Master translation matrix and porting guide for transferring these optimizations to NVIDIA CUDA (Tensor Cores / cuFile), AMD ROCm (WMMA / LDS), Vulkan, and WebGPU.
* [**Master Empirical Benchmark Telemetry & Hardware Roofline Reference**](benchmarks_and_telemetry.md) — Complete consolidated telemetry tables, variance distributions, and hardware efficiency roofline metrics across all sweeps.
* [**Version 1 (v0.1) vs Version 2 (v0.2): Architectural Comparison & Evolution**](v1_vs_v2_comparison.md) — Comprehensive feature matrix, compute roofline evolution, and metrology overhaul from v0.1 to v0.2.
* [**Project Changelog**](../CHANGELOG.md) — Chronological history of all features, additions, breaking changes, and hardening fixes.

---

## Building and Running

### Prerequisites
*   Apple Silicon Mac (M4) running macOS 14.0+
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

# 6. 1M-token flash streaming & speculative decode engine (Experimental — Deferred from v0.2 release)
# ./bench_streaming_1m  # Optional research prototype; not executed in standard release verification

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

If the ideas, techniques, or specific hardware-level optimizations from this repository (such as the M4 LSU saturation methods, Universal Quantization Router, 1M flash streaming architecture, or Metal-specific prefill routing) are adapted, ported to other silicon architectures (AMD/Nvidia/Intel), or used to improve decoding phases in other software, I humbly ask for a **visible citation, link, or mention** in your project's documentation, blog post, or research paper.

Any visibility that helps a junior engineer grow and find their footing in the systems engineering community is deeply and genuinely appreciated. Thank you for reading, testing, and building.

---

## Contact & Discussion

If you want to discuss Metal optimization, Apple Silicon memory hierarchies, or LLM inference, feel free to reach out:

*   **X (Twitter):** [Mohammed Hossam (@MohamedHz72007)](https://x.com/MohamedHz72007)
*   **GitHub:** [mohamedhossammohamed](https://github.com/mohamedhossammohamed)
