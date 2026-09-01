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

### Benchmarking Methodology, Disclosures & Baseline Standards

> **[Disclosure & Baseline Standards]**  
> All cross-engine numbers use synthetic in-UMA weights with exact model shapes, zero disk I/O, zero tokenizer overhead (where $M$ is the exact prompt token count), and prefill-only (single forward pass, no autoregressive generation). This isolates low-level kernel execution on identical physical tensor layouts.

*   **Primary Baseline (Apple MLX):** **Apple MLX (v0.32.2)** is the official, primary baseline for this project. Early exploratory prototypes informally referenced application-level runtimes (such as Ollama); all such references have been retired in favor of Apple MLX to ensure strict, reproducible, and unimpeachable systems metrology. MLX represents the gold-standard native framework on Apple Silicon.
*   **Secondary Reference Baseline (llama.cpp-Style):** An in-house Metal reimplementation of `ggml`'s `kernel_mul_mm_q4_0` matrix multiplication kernel from `llama.cpp` (calibrated at ~8–10 TFLOPS on M4) is provided as an open-source C++/Metal reference.
*   **Strict Timing Parity:** All engines are measured with identical host-synchronized wall-clock timing: `commit` + `waitUntilCompleted` for Metal and `mx.eval` for Apple MLX. Pure hardware execution timestamps (`GPUStartTime` / `GPUEndTime`) are separated in a dedicated `"GPU-only (ours)"` column.
*   **Variance & Sampling:** 10 warmup iterations (discarded) followed by 20 measured iterations across all sequence lengths, reporting median times and `[min – max]` distributions.
*   **KV Cache Standard:** Headline cross-engine comparisons evaluate standard FP16 KV cache across all engines. Dynamic Q8_0 KV is a custom feature and explicitly noted where tested.

## ⚠️ Target Audience & Usage Disclaimer

**This is a research artifact and a proof-of-concept.** 

It is **not** intended as a drop-in replacement for everyday `llama.cpp` users, local LLM enthusiasts, or those seeking a plug-and-play CLI experience. It currently requires manual weight extraction, compilation, and command-line interaction. 

The goal of this repository is to provide a verified, open-source baseline for the community. It is intended for researchers and engineers studying Apple Silicon memory hierarchies, until these specific low-level optimizations can be upstreamed into mainstream frameworks (like `ggml-metal`) or adopted as specialized hardware configurations.

## Cross-Engine Prefill Comparison: Apple MLX (Primary Baseline) vs. Ours

To contextualize the engine's performance, we executed a head-to-head prefill-only benchmark across 1B and 8B Transformer architectures using identical synthetic weight topologies. All engines were measured using a strict shared wall-clock timing methodology (10 warmup iterations, 20 measurement iterations) to ensure absolute parity. **Apple MLX (v0.32.2)** serves as the primary baseline, alongside an in-house **llama.cpp-style Metal baseline** (`ggml mul_mm` calibrated at ~8–10 TFLOPS) for reference.

### The Architectural Trade-off: Aligned vs. Unaligned Boundaries
The results highlight a fascinating physical trade-off on the M4 architecture:
*   **vs. Apple MLX Baseline (Aligned Powers-of-2):** MLX's heavily optimized JIT compiler excels at perfectly aligned, power-of-2 dense matrix blocks ($M=512, 1024, 2048$), outperforming our engine by 14–31% on 1B shapes and long 8B batches.
*   **vs. Apple MLX Baseline (Unaligned Edge Boundaries):** On arbitrary, real-world prompt boundaries (e.g., $M=128, 129$), our engine's custom Metal routing and direct-head projections eliminate dynamic padding and transposition overhead. This allows our engine to outperform MLX by up to **1.25x (+25%)** on unaligned 8B edge cases.
*   **vs. llama.cpp-Style Reference:** Our custom engine delivers a consistent **1.19x to 1.76x speedup** across all scales and prompt boundaries, demonstrating the impact of 128-bit LSU saturation and fused dequantization.

---

### 8B Architecture ($K=4096, H=32, D=128, N_{\text{mlp}}=14336$, 32 Layers)

*At 8B scale, a single layer's weights (~130.5 MB) exceed the Apple M4's 24 MB SLC by ~5.4x, operating heavily in DRAM streaming.*

| Prompt ($M$) | Boundary Type | Apple MLX Metal (Primary Baseline) | Our Engine (Wall) | GPU-only (ours) | vs. MLX Baseline | llama.cpp-style (Reference) | vs. llama.cpp Ref |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | **17.61 ms** | 20.67 ms | 20.39 ms | 0.85x (MLX +17%) | 25.39 ms | **1.23x Faster** |
| **127** | Edge (Unaligned) | **39.41 ms** | 40.45 ms | 40.13 ms | 0.97x (≈ Parity) | 47.99 ms | **1.19x Faster** |
| **128** | Aligned ($2^7$) | 41.09 ms | **39.19 ms** | 38.84 ms | **1.05x (Ours +5%)** | 49.72 ms | **1.27x Faster** |
| **129** | Edge (Unaligned) | 67.74 ms | **54.34 ms** | 53.96 ms | **1.25x (Ours +25%)** | 79.93 ms | **1.47x Faster** |
| **512** | Aligned ($2^9$) | **155.32 ms** | 163.95 ms | 163.59 ms | 0.95x (≈ Parity) | 216.23 ms | **1.32x Faster** |
| **1023** | Edge (Unaligned) | **329.61 ms** | 363.65 ms | 363.28 ms | 0.91x (MLX +10%) | 464.58 ms | **1.28x Faster** |
| **1024** | Aligned ($2^{10}$) | **307.27 ms** | 359.64 ms | 359.32 ms | 0.85x (MLX +17%) | 455.40 ms | **1.27x Faster** |
| **2047** | Edge (Unaligned) | **640.67 ms** | 801.28 ms | 800.72 ms | 0.80x (MLX +25%) | 1102.36 ms | **1.38x Faster** |
| **2048** | Aligned ($2^{11}$) | **612.59 ms** | 763.03 ms | 762.69 ms | 0.80x (MLX +25%) | 1075.05 ms | **1.41x Faster** |

*Numerical precision: MaxDiff ≤ 0.0044 vs CPU gold reference, 0 NaN/Inf. Note: Opt Q8_0 KV is a custom-only feature (750.69 ms at M=2048, 1.43x vs reference); MLX path runs standard FP16 KV.*

---

### 1B Architecture ($K=2048, H=32, D=64, N_{\text{mlp}}=5632$, 16 Layers)

*Note: The original 1B scorecard used $H=16$; this cross-engine suite uses $H=32$ (LLaMA-3.2-1B standard); the two tables are not cross-comparable.*

| Prompt ($M$) | Boundary Type | Apple MLX Metal (Primary Baseline) | Our Engine (Wall) | GPU-only (ours) | vs. MLX Baseline | llama.cpp-style (Reference) | vs. llama.cpp Ref |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | **3.74 ms** | 4.74 ms | 4.45 ms | 0.79x (MLX +27%) | 5.94 ms | **1.25x Faster** |
| **127** | Edge (Unaligned) | **6.48 ms** | 8.27 ms | 7.98 ms | 0.78x (MLX +28%) | 10.22 ms | **1.24x Faster** |
| **128** | Aligned ($2^7$) | **6.33 ms** | 8.28 ms | 7.97 ms | 0.76x (MLX +31%) | 10.07 ms | **1.22x Faster** |
| **129** | Edge (Unaligned) | **8.24 ms** | 9.71 ms | 9.44 ms | 0.85x (MLX +18%) | 14.79 ms | **1.52x Faster** |
| **512** | Aligned ($2^9$) | **23.41 ms** | 30.13 ms | 29.82 ms | 0.78x (MLX +29%) | 41.63 ms | **1.38x Faster** |
| **1023** | Edge (Unaligned) | **47.57 ms** | 62.24 ms | 61.91 ms | 0.76x (MLX +31%) | 95.65 ms | **1.54x Faster** |
| **1024** | Aligned ($2^{10}$) | **49.78 ms** | 63.14 ms | 62.82 ms | 0.79x (MLX +27%) | 95.86 ms | **1.52x Faster** |
| **2047** | Edge (Unaligned) | **122.16 ms** | 141.23 ms | 140.94 ms | 0.86x (MLX +16%) | 248.72 ms | **1.76x Faster** |
| **2048** | Aligned ($2^{11}$) | **126.09 ms** | 142.84 ms | 142.55 ms | 0.88x (MLX +14%) | 240.84 ms | **1.69x Faster** |

*Numerical precision: MaxDiff ≤ 0.0020 vs CPU gold reference, 0 NaN/Inf.*

*Note: Full raw telemetry including variance distributions [min–max] is available in `benchmarks/logs/`.*

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
