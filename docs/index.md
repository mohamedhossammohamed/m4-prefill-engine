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

## Benchmarking Methodology
To ensure accuracy, all latency measurements in this repository are captured using native Metal GPU execution timestamps (`GPUStartTime` / `GPUEndTime`), isolating pure kernel compute time from CPU driver dispatch overhead. 

The test suite rigorously validates numerical stability across non-aligned and edge-case sequence boundaries (M ∈ [33, 127, 128, 129, 512, 1023, 1024, 2047, 2048]) to ensure causal masking and thread-group guards hold under stress, and verifies sustained thermal equilibrium under continuous 60-second loads.

## ⚠️ Target Audience & Usage Disclaimer

**This is a research artifact and a proof-of-concept.** 

It is **not** intended as a drop-in replacement for everyday `llama.cpp` users, local LLM enthusiasts, or those seeking a plug-and-play CLI experience. It currently requires manual weight extraction, compilation, and command-line interaction. 

The goal of this repository is to provide a verified, open-source baseline for the community. It is intended for researchers and engineers studying Apple Silicon memory hierarchies, until these specific low-level optimizations can be upstreamed into mainstream frameworks (like `ggml-metal`) or adopted as specialized hardware configurations.

## Scaling Check: 8B Architecture (LLaMA-3 8B Shapes)

The 1B model results operate in a regime where kernel fusion, direct-head layout routing, and Load-Store Unit (LSU) saturation dominate execution time. At 8B scale ($K=4096, H=32, D=128, N_{\text{mlp}}=14336$, 32 layers, 4.38 GB in Q4_0), a single layer's weights (~137 MB) exceed the Apple M4's 24 MB System-Level Cache (SLC) by ~5.7x, forcing continuous DRAM streaming across the unified memory bus—the exact operational regime where generational hardware memory bandwidth jumps (such as M5) pay off.

Software optimizations and hardware bandwidth improvements are complementary: this engine's fusion, transpose elimination, and 128-bit LSU queue saturation techniques deliver substantial, measurable gains in this heavy memory-streaming regime.

### Head-to-Head 8B Prefill (Apple M4, GPU Timestamps)

| Sequence Length (M) | Baseline (GPU ms/layer) | This Engine (GPU ms/layer) | Speedup | Full 32-Layer Time (Ours) |
| :---: | :---: | :---: | :---: | :---: |
| **M = 33** (Edge) | 22.37 ms | **18.12 ms** | **1.23x** | 579.7 ms |
| **M = 127** (Edge) | 44.90 ms | **35.32 ms** | **1.27x** | 1.13 s |
| **M = 128** | 43.84 ms | **35.31 ms** | **1.24x** | 1.13 s |
| **M = 129** (Edge) | 63.11 ms | **43.26 ms** | **1.46x** | 1.38 s |
| **M = 512** | 181.84 ms | **142.23 ms** | **1.28x** | 4.55 s |
| **M = 1023** (Edge) | 409.64 ms | **305.70 ms** | **1.34x** | 9.78 s |
| **M = 1024** | 414.36 ms | **312.42 ms** | **1.33x** | 10.00 s |
| **M = 2047** (Edge) | 923.20 ms | **657.96 ms** | **1.40x** | 21.05 s |
| **M = 2048** | 910.48 ms | **688.18 ms** | **1.32x** | 22.02 s |

*Numerical precision: MaxDiff = 0.02344 vs CPU gold reference, 0 NaN/Inf.*

> **8B Scaling Highlight:**  
> A 2048-token prompt across all 32 layers drops from **29.14 s to 22.02 s — 7.11 seconds saved per prompt**, in the DRAM-streaming regime where hardware bandwidth normally dominates.

### Component Breakdown (M = 2048, 1 Layer)

| Subsystem | Baseline (GPU ms) | This Engine (GPU ms) | Speedup |
| :--- | :---: | :---: | :---: |
| **QKV Projections** | 152.86 ms | **100.18 ms** | **1.55x** |
| **Causal FlashAttention** | 162.55 ms | **103.87 ms** | **1.56x** |
| **Output Projection** | 50.96 ms | **32.90 ms** | **1.60x** |
| **MLP (SwiGLU)** | 544.11 ms | **451.23 ms** | **1.21x** |

At 8B scale, the measured arithmetic intensity is ≈ 7,800 FLOPs/byte (firmly compute-bound against the M4 roofline ridge). The current gains stem directly from kernel fusion, transpose elimination, and 128-bit LSU saturation; dedicated 8B tile-geometry tuning represents active ongoing work with substantial headroom. Deep software optimization and next-generation hardware bandwidth scaling stack multiplicatively.

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
