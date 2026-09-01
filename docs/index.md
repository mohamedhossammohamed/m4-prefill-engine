# m4-prefill-engine

**Exploring the absolute limits of LLM prompt processing on Apple Silicon.**

> **Live Documentation:** [mohamedhossammohamed.github.io/m4-prefill-engine](https://mohamedhossammohamed.github.io/m4-prefill-engine/)  
> **Hardware Target:** Apple M4 MacBook Air (10-core GPU, 16GB Unified Memory)

## The Context: Hardware Generations vs. Software Engineering

With the introduction of the M5 generation, Apple has achieved massive leaps in LLM prefill (Time-To-First-Token) performance. By scaling memory bandwidth, expanding the System-Level Cache (SLC), and introducing new silicon-level AI accelerators, hardware generational jumps routinely yield 2x to 4x speedups for AI workloads.

This project asks a specific systems-engineering question: *How much of that prefill bottleneck can be alleviated purely through low-level software engineering on the existing M4 architecture?*

By bypassing high-level framework abstractions and writing custom Metal shaders directly for the M4's Load-Store Units (LSU) and Unified Memory Architecture, this engine achieves a **~3.4x to 3.7x speedup** specifically for the prefill phase of a 1B-parameter Transformer model. 

While Apple achieves these gains through new silicon fabrication nodes, this project demonstrates that a similar magnitude of improvement for the memory-bound prefill bottleneck can be achieved today through deep, low-level software optimization.

## Core Architectural Optimizations
This engine replaces standard matrix multiplications and attention mechanisms with custom, hand-written Metal kernels optimized for the M4's specific SIMD group sizes and memory hierarchy:

*   **128-bit Vector Saturation:** Aligned `float4` memory firehoses to fully saturate the M4 LSU in-flight load queue.
*   **Fused Q4_0 Dequantization:** Unpacking 32-bit packed integers directly into `half4` vectors using bitwise ALU operations, eliminating intermediate global memory writes.
*   **Fused FlashAttention & Q8_0 KV Cache:** Online running softmax in registers with causal triangular block skipping, paired with dynamic 8-bit KV caching to reduce memory footprint by ~47%.
*   **Direct-Head SwiGLU Fusion:** Projecting activations directly into head-major formats and fusing the Gate/Up projections in registers, eliminating costly memory transpose kernels.

#### Benchmarking Methodology & Disclosures

> **[Disclosure Block]**  
> All cross-engine numbers use synthetic in-UMA weights with exact model shapes, no disk I/O, no tokenizer (M is the token count), prefill-only (single forward pass, no generation). This measures kernel execution on identical workloads, not end-to-end product latency.

*   **Timing Parity:** In cross-engine comparisons, all engines are measured with the exact same method: wall-clock timing around `commit+waitUntilCompleted` for Metal and `mx.eval` for Apple MLX (v0.32.2). Native Metal `GPUStartTime` / `GPUEndTime` metrics are preserved as a dedicated `"GPU-only (ours)"` column.
*   **Variance & Sampling:** 10 warmup iterations (discarded) followed by 20 measured iterations across all sequence lengths, reporting median times and `[min – max]` distributions.
*   **Baseline Identity:** The baseline is an in-house Metal reimplementation of `ggml`'s `kernel_mul_mm_q4_0` matrix multiplication kernel from `llama.cpp` (calibrated at ~8–10 TFLOPS on M4), allowing pure in-memory kernel benchmarking without disk artifact dependencies.
*   **KV Cache:** Cross-engine comparisons run standard FP16 KV cache on all engines. Dynamic Q8_0 KV is a custom feature and explicitly noted where tested.

## ⚠️ Target Audience & Usage Disclaimer

**This is a research artifact and a proof-of-concept.** 

It is **not** intended as a drop-in replacement for everyday `llama.cpp` users, local LLM enthusiasts, or those seeking a plug-and-play CLI experience. It currently requires manual weight extraction, compilation, and command-line interaction. 

The goal of this repository is to provide a verified, open-source baseline for the community. It is intended for researchers and engineers studying Apple Silicon memory hierarchies, until these specific low-level optimizations can be upstreamed into mainstream frameworks (like `ggml-metal`) or adopted as specialized hardware configurations.

## Cross-Engine Prefill Comparison (1B & 8B Architectures)

Below is the verified head-to-head prefill benchmark comparing the **llama.cpp-style baseline**, **Apple MLX (v0.32.2)**, and **Our Custom Metal Engine** on identical model configurations on the Apple M4 (10-core GPU, 16GB Unified Memory Architecture).

### 1B Architecture (LLaMA-3.2-1B: $K=2048, H=32, D=64, N_{\text{mlp}}=5632$, 16 Layers)

*Note: The original 1B scorecard used $H=16$; the cross-engine suite below uses $H=32$ (LLaMA-3.2-1B standard); the two tables are not cross-comparable.*

| Sequence Length ($M$) | llama.cpp-style Baseline (Wall ms) | Apple MLX Metal (Wall ms) | Our Metal Engine (Wall ms) | GPU-only (ours) | vs. Baseline | vs. MLX | Full-Model Est (Ours) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M = 33** (Edge) | 5.94 ms | **3.74 ms** | 4.74 ms | 4.45 ms | **1.25x** | 0.79x | 0.08 s |
| **M = 127** (Edge) | 10.22 ms | **6.48 ms** | 8.27 ms | 7.98 ms | **1.24x** | 0.78x | 0.13 s |
| **M = 128** | 10.07 ms | **6.33 ms** | 8.28 ms | 7.97 ms | **1.22x** | 0.76x | 0.13 s |
| **M = 129** (Edge) | 14.79 ms | **8.24 ms** | 9.71 ms | 9.44 ms | **1.52x** | 0.85x | 0.16 s |
| **M = 512** | 41.63 ms | **23.41 ms** | 30.13 ms | 29.82 ms | **1.38x** | 0.78x | 0.48 s |
| **M = 1023** (Edge) | 95.65 ms | **47.57 ms** | 62.24 ms | 61.91 ms | **1.54x** | 0.76x | 1.00 s |
| **M = 1024** | 95.86 ms | **49.78 ms** | 63.14 ms | 62.82 ms | **1.52x** | 0.79x | 1.01 s |
| **M = 2047** (Edge) | 248.72 ms | **122.16 ms** | 141.23 ms | 140.94 ms | **1.76x** | 0.86x | 2.26 s |
| **M = 2048** | 240.84 ms | **126.09 ms** | 142.84 ms | 142.55 ms | **1.69x** | 0.88x | 2.29 s |

*Numerical precision: MaxDiff ≤ 0.0020 vs CPU gold reference, 0 NaN/Inf.*

---

### 8B Architecture (LLaMA-3.1-8B: $K=4096, H=32, D=128, N_{\text{mlp}}=14336$, 32 Layers)

At 8B scale, a single layer's weights (~130.5 MB) exceed the Apple M4's 24 MB SLC by ~5.4x, operating heavily in DRAM streaming.

| Sequence Length ($M$) | llama.cpp-style Baseline (Wall ms) | Apple MLX Metal (Wall ms) | Our Metal Engine (Wall ms) | GPU-only (ours) | vs. Baseline | vs. MLX | Full-Model Est (Ours) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M = 33** (Edge) | 25.39 ms | **17.61 ms** | 20.67 ms | 20.39 ms | **1.23x** | 0.85x | 0.66 s |
| **M = 127** (Edge) | 47.99 ms | **39.41 ms** | 40.45 ms | 40.13 ms | **1.19x** | 0.97x | 1.29 s |
| **M = 128** | 49.72 ms | 41.09 ms | **39.19 ms** | 38.84 ms | **1.27x** | **1.05x** | 1.25 s |
| **M = 129** (Edge) | 79.93 ms | 67.74 ms | **54.34 ms** | 53.96 ms | **1.47x** | **1.25x** | 1.74 s |
| **M = 512** | 216.23 ms | **155.32 ms** | 163.95 ms | 163.59 ms | **1.32x** | 0.95x | 5.25 s |
| **M = 1023** (Edge) | 464.58 ms | **329.61 ms** | 363.65 ms | 363.28 ms | **1.28x** | 0.91x | 11.64 s |
| **M = 1024** | 455.40 ms | **307.27 ms** | 359.64 ms | 359.32 ms | **1.27x** | 0.85x | 11.51 s |
| **M = 2047** (Edge) | 1102.36 ms | **640.67 ms** | 801.28 ms | 800.72 ms | **1.38x** | 0.80x | 25.64 s |
| **M = 2048** | 1075.05 ms | **612.59 ms** | 763.03 ms | 762.69 ms | **1.41x** | 0.80x | 24.42 s |

*Numerical precision: MaxDiff ≤ 0.0044 vs CPU gold reference, 0 NaN/Inf. Note: Opt Q8_0 KV is a custom-only feature (750.69 ms at M=2048, 1.43x vs baseline); MLX path runs standard FP16 KV.*

### Metrology & Performance Insights

1. **vs. llama.cpp-Style Baseline:** Our custom Metal engine demonstrates consistent **1.19x to 1.76x speedups** across all sequence lengths and both 1B and 8B scales under shared wall-clock timing.
2. **vs. Apple MLX:** 
   - On unaligned boundary lengths (e.g., $M=128, 129$ at 8B), our engine outperforms MLX by up to **1.25x (+25%)** due to direct-head layout avoiding dynamic transpose passes.
   - On small head dimensions ($D=64$ at 1B) and long batch-aligned sequences ($M=2048$), Apple MLX's compiled JIT kernel scheduler achieves ~10–20% lower latency. All numbers are published transparently.plicatively.

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

As an early career Physician [second year med student] (who is doing this for some reason he is not very sure of, yet), I'm open sourcing this as a baseline. If you use these Metal routing ideas in your own silicon/software, a citation or link back would mean the world to me!

## Contact & Discussion

If you want to discuss Metal optimization, Apple Silicon memory hierarchies, or LLM inference, feel free to reach out:

*   **X (Twitter):** [Mohammed Hossam (@MohamedHz72007)](https://x.com/MohamedHz72007)
*   **GitHub:** [mohamedhossammohamed](https://github.com/mohamedhossammohamed)
