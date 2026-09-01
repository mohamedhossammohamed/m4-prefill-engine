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

## Cross-Engine Prefill Comparison (llama.cpp-style vs. Apple MLX vs. Ours)

To contextualize the engine's performance, we executed a head-to-head prefill-only benchmark across 1B and 8B Transformer architectures using identical synthetic weight topologies. All engines were measured using a strict shared wall-clock timing methodology (10 warmup iterations, 20 measurement iterations) to ensure absolute parity.

### The Architectural Trade-off: Aligned vs. Unaligned Boundaries
The results highlight a fascinating physical trade-off on the M4 architecture:
*   **vs. llama.cpp-style baseline:** Our engine delivers a consistent **1.19x to 1.76x speedup** across all scales and sequence lengths, proving the efficacy of 128-bit LSU saturation and fused dequantization.
*   **vs. Apple MLX (Aligned):** MLX's heavily optimized JIT compiler excels at perfectly aligned, power-of-2 dense matrix blocks, outperforming our engine by 14–31% on standard 1B shapes and long 8B batches.
*   **vs. Apple MLX (Unaligned):** On arbitrary, real-world prompt boundaries (e.g., $M=129$, $M=1023$), our engine's custom Metal routing and direct-head projections eliminate padding and transposition penalties. This allows our engine to outperform MLX by up to **1.25x (+25%)** on unaligned 8B edge cases.

#### 8B Architecture ($K=4096, H=32, D=128$, 32 Layers) - Selected Highlights
| Prompt ($M$) | llama.cpp-style Baseline | Apple MLX Metal | Our Engine (Wall) | vs. Baseline | vs. MLX |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **128 (Aligned)** | 49.72 ms | **41.09 ms** | 39.19 ms | 1.27x Faster | **1.05x (Ours +5%)** |
| **129 (Edge)** | 79.93 ms | 67.74 ms | **54.34 ms** | 1.47x Faster | **1.25x (Ours +25%)** |
| **512 (Aligned)** | 216.23 ms | **155.32 ms** | 163.95 ms | 1.32x Faster | 0.95x (Parity) |
| **2048 (Aligned)** | 1075.05 ms | **612.59 ms** | 763.03 ms | 1.41x Faster | 0.80x (MLX +25%) |

*Note: Full 1B and 8B telemetry, including variance bounds [min-max] and GPU-only timestamp breakdowns, is available in the `benchmarks/logs/` directory. The llama.cpp-style baseline is an in-house Metal reimplementation of `ggml mul_mm`, calibrated to ~8-10 TFLOPS on M4.*

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
