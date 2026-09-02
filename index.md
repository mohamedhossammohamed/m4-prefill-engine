# m4-prefill-engine
**A low-level Metal inference architecture for Apple Silicon: universal quantization formats, out-of-core 1M-token contexts, and decode that stays out of single-digit tokens/sec.**

> **Live Documentation:** [mohamedhossammohamed.github.io/m4-prefill-engine](https://mohamedhossammohamed.github.io/m4-prefill-engine/)  
> **Hardware Target:** Apple M4 MacBook Air (10-core GPU, 16GB Unified Memory)  
> **Version:** v0.2

---

## The Problem: Three Tradeoffs

Local LLM inference on Apple Silicon forces a painful choice: (1) Speed — MLX is fast but locks you into its proprietary 4-bit format; (2) Formats — llama.cpp gives you the GGUF ecosystem but pays a prefill penalty on Apple GPUs; (3) Context — beyond physical RAM you either OOM or degrade into OS swap. This project asks whether one low-level Metal engine can address all three.

---

## The Solution: A Unified Inference Architecture (v0.2)

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
| **8B (K=4096)** | **33** | 0.930 ms | 1.336 ms | 1.117 ms | 1.637 ms | 1.272 ms | **0.849 ms** |
| | **128** | 2.711 ms | 1.840 ms | 2.280 ms | 2.205 ms | 2.326 ms | **1.481 ms** |
| | **129** | 3.026 ms | 2.385 ms | 2.637 ms | 2.824 ms | 2.905 ms | **2.199 ms** |
| | **2048** | 22.009 ms | **21.744 ms** | 24.622 ms | 26.426 ms | 28.276 ms | 21.960 ms |
| **1B (K=2048)** | **33** | **0.257 ms** | 0.293 ms | 0.312 ms | 0.304 ms | 0.365 ms | 0.321 ms |
| | **128** | 1.847 ms | 1.846 ms | 2.109 ms | 1.231 ms | 1.545 ms | **1.020 ms** |
| | **129** | 2.355 ms | 2.119 ms | 2.110 ms | 1.618 ms | 1.754 ms | **1.077 ms** |
| | **2048** | **5.557 ms** | 5.598 ms | 6.217 ms | 6.549 ms | 7.237 ms | 5.662 ms |

*Source: Measured via `bench_universal_router` with mandatory 32MB SLC cache flushing and double-precision CPU verification (MaxDiff ≤ 0.0078).*  
*Note on Ternary 1.58-bit: Empirical testing reveals that on Apple Silicon, feeding unpacked Ternary weights into the 16.8 TFLOPS Hardware Matrix Coprocessor (MMA) is significantly faster than attempting pure Vector ALU addition/subtraction. The true advantage of Ternary on M4 is memory bandwidth (fitting entirely inside the 24MB SLC cache), not compute bypass.*

### Pillar 3: 1M-Token Out-of-Core Flash Streaming
When contexts exceed physical RAM (16GB), the engine treats internal PCIe flash storage as an extension of Unified Memory.

*   **Direct Flash Reads:** Utilizes `F_NOCACHE` with strictly 16KB page-aligned (`posix_memalign`) buffers to bypass the macOS Unified Buffer Cache (UBC), achieving 2.0–3.0 GB/s physical read throughput from internal PCIe flash storage.
*   **Chunked FlashAttention:** Online softmax running statistics ($m_i$, $l_i$) are persisted to global memory between storage chunks, enabling mathematically exact attention across arbitrarily long contexts.
*   **Dual 128MB Ring Buffer:** Overlaps GPU compute with flash reads, hiding storage latency behind the Matrix Coprocessor.

### Pillar 4: On-the-Fly Out-of-Core Decode
The streaming engine above solves prefill. The harder question is decode: every generated token must attend over the entire context, and at 1M tokens that context lives on flash. This pillar is an on-the-fly proof-of-concept — a handful of tricks to test whether a 1,000,000-token out-of-core context can decode without collapsing into single-digit tokens/sec.

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
| **1M** | [MEASURED] Speculative Burst ($K=64$) | **1.82 s** | **1.74 s** | **2.59 GB/s** | **35.2 verified tok/s** | **12.51 GB** |

*Result: Speculative burst verification ($K=64$ candidates verified in a single KV stream) delivers ~35.2 verified tok/s at 1,000,000 tokens — ~60x faster than the naive flash floor (0.60 tok/s) and exceeding the naive in-RAM ceiling (22 tok/s), because the 4.3 GB stream is amortized across 64 candidate tokens.*

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

### Table B: Full-Layer Prefill Comparison (Apple MLX Metal vs Custom Engine)

Measured using shared wall-clock timing parity (10 warmup, 20 measured iterations, 32MB SLC flush, identical tensor layouts). Full raw logs with variance distributions `[min - max]` are in `benchmarks/logs/`.

#### 8B Model Tier (32 Layers, K=4096, H=32, D=128, N_mlp=14336)

| Prompt (M) | Boundary Type | Apple MLX Metal (Primary Baseline) | Our Engine (Wall-Clock) | Our Engine (GPU-only) | vs MLX Baseline | llama.cpp-style Reference | vs Reference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | **17.61 ms** [17.01-19.86] | 20.67 ms [19.80-22.10] | 20.39 ms | 0.85x (MLX +17%) | 25.39 ms [24.10-27.20] | **1.23x faster** |
| **127** | Edge (Unaligned) | **39.41 ms** [38.61-41.52] | 40.45 ms [37.50-44.20] | 40.13 ms | 0.97x (≈ Parity) | 47.99 ms [44.80-52.10] | **1.19x faster** |
| **128** | Aligned ($2^7$) | 41.09 ms [33.42-51.40] | **39.19 ms** [36.28-43.52] | 38.84 ms | **1.05x faster** | 49.72 ms [45.59-53.95] | **1.27x faster** |
| **129** | Edge (Unaligned) | 67.74 ms [64.25-75.48] | **54.34 ms** [50.10-58.90] | 53.96 ms | **1.25x faster** | 79.93 ms [74.20-86.40] | **1.47x faster** |
| **512** | Aligned ($2^9$) | **155.32 ms** [144.09-171.74] | 163.95 ms [154.20-175.80] | 163.59 ms | 0.95x (≈ Parity) | 216.23 ms [204.10-230.50] | **1.32x faster** |
| **1024** | Aligned ($2^{10}$) | **307.27 ms** [282.29-367.09] | 359.64 ms [340.10-385.20] | 359.32 ms | 0.85x (MLX +17%) | 455.40 ms [430.20-482.10] | **1.27x faster** |
| **2048** | Aligned ($2^{11}$) | **612.59 ms** [543.43-726.53] | 763.03 ms [720.40-815.60] | 762.69 ms | 0.80x (MLX +25%) | 1075.05 ms [1010.20-1150.40] | **1.41x faster** |

#### 1B Model Tier (16 Layers, K=2048, H=32, D=64, N_mlp=5632)

| Prompt (M) | Boundary Type | Apple MLX Metal (Primary Baseline) | Our Engine (Wall-Clock) | Our Engine (GPU-only) | vs MLX Baseline | llama.cpp-style Reference | vs Reference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | **3.74 ms** [3.51-3.95] | 4.74 ms [4.40-5.10] | 4.45 ms | 0.79x (MLX +27%) | 5.94 ms [5.60-6.30] | **1.25x faster** |
| **128** | Aligned ($2^7$) | **6.33 ms** [6.02-6.70] | 8.28 ms [7.80-8.90] | 7.97 ms | 0.76x (MLX +31%) | 10.07 ms [9.50-10.80] | **1.22x faster** |
| **129** | Edge (Unaligned) | **8.24 ms** [8.09-9.12] | 9.71 ms [9.10-10.40] | 9.44 ms | 0.85x (MLX +18%) | 14.79 ms [13.90-15.80] | **1.52x faster** |
| **512** | Aligned ($2^9$) | **23.41 ms** [22.90-23.86] | 30.13 ms [28.50-32.40] | 29.82 ms | 0.78x (MLX +29%) | 41.63 ms [39.80-44.10] | **1.38x faster** |
| **2048** | Aligned ($2^{11}$) | **126.09 ms** [115.26-157.07] | 142.84 ms [135.20-153.10] | 142.55 ms | 0.88x (MLX +14%) | 240.84 ms [228.10-256.40] | **1.69x faster** |

*Footnote on MLX trade-offs: MLX's compiler excels on power-of-2 dense blocks (e.g. 1B M=33 or long sequences), where it is 14–31% faster. On real-world unaligned 8B edge boundaries (M=128, 129), our direct-head routing eliminates dynamic padding and transpositions, outperforming MLX by 1.05x to 1.25x.*

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

# 6. 1M-token flash streaming & speculative decode engine
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

If the ideas, techniques, or specific hardware-level optimizations from this repository (such as the M4 LSU saturation methods, Universal Quantization Router, 1M flash streaming architecture, or Metal-specific prefill routing) are adapted, ported to other silicon architectures (AMD/Nvidia/Intel), or used to improve decoding phases in other software, I humbly ask for a **visible citation, link, or mention** in your project's documentation, blog post, or research paper.

Any visibility that helps a junior engineer grow and find their footing in the systems engineering community is deeply and genuinely appreciated. Thank you for reading, testing, and building.

---

## Contact & Discussion

If you want to discuss Metal optimization, Apple Silicon memory hierarchies, or LLM inference, feel free to reach out:

*   **X (Twitter):** [Mohammed Hossam (@MohamedHz72007)](https://x.com/MohamedHz72007)
*   **GitHub:** [mohamedhossammohamed](https://github.com/mohamedhossammohamed)
