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

If the ideas, techniques, or specific hardware-level optimizations from this repository (such as the M4 LSU saturation methods, custom Q8_0 KV cache handling, or Metal-specific prefill routing) are adapted, ported to other silicon architectures (AMD/Nvidia/Intel), or used to improve decoding phases in other software, I humbly ask for a **visible citation, link, or mention** in your project's documentation, blog post, or research paper. 

Any visibility that helps a junior engineer grow and find their footing in the systems engineering community is deeply and genuinely appreciated. Thank you for reading, testing, and building.

## Contact & Discussion

If you want to discuss Metal optimization, Apple Silicon memory hierarchies, or LLM inference, feel free to reach out:

*   **X (Twitter):** [Mohammed Hossam (@MohamedHz72007)](https://x.com/MohamedHz72007)
*   **GitHub:** [mohamedhossammohamed](https://github.com/mohamedhossammohamed)
