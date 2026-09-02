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

---

## M3 Ultra Support & Tuning Notes

This section documents a port of the 8B prefill path to the **Apple M3 Ultra** (80-core GPU,
256 GB unified, macOS 26.5.1, Metal 4). The M4 path is unchanged and still selectable.

### Why the M4 tuning does not transfer

The upstream kernels are tuned for a chip where memory is the binding constraint. On an
M3 Ultra it is not, and the measurements are not close:

| | measured on this machine |
|---|---|
| memory bandwidth ceiling, read-only | **776 GB/s** (95% of 819.2 theoretical) |
| memory bandwidth ceiling, read+write | **695 GB/s** (85%) |
| bandwidth the engine actually used | **1.5 GB/s — 0.2% of it** |
| FP16 / Q4 GEMM reference (MLX, same shapes) | **24.8 TFLOP/s** |
| upstream causal attention | **2.0 TFLOP/s — 8% of that** |
| upstream fused MLP | 10.5 TFLOP/s — 42% |

Every kernel was compute- and occupancy-limited, not bandwidth-limited. The specific cause
is that all GEMM and attention kernels dispatch **32-thread threadgroups** (one SIMD group)
and give one output column, or one query row, to a single thread. In attention that costs
~128 registers per thread and 16 KB of threadgroup memory, so at M=2048 the dispatch is
65,536 threads against ~80,000 thread slots: the machine fills exactly once, with no
latency hiding.

### What changed

All four stages moved to `simdgroup_matrix` with 128-thread (4 simdgroup) threadgroups:

| new kernel | replaces | note |
|---|---|---|
| `flash_attn_sg_causal_d128` | `flash_attn_fp16_causal_d128` | 8 query rows per simdgroup, float O accumulators |
| `flash_attn_sg_q8_0_causal_d128` | `flash_attn_q8_0_causal_d128` | same body, different tile loader |
| `sg_gemm_q4_0` | `pipe_gemm_q4_0_32x32` | BN=64, BK=32 (one q4_0 block per step) |
| `sg_gemm_q4_0_fused_swiglu` | `fused_gate_up_swiglu_q4_0` | 32-row tile; see the fusion note below |
| `sg_qkv_head_gemm_q4_0` | `pipe_qkv_head_gemm_q4_0` | same body with a `[H, M, D]` scatter epilogue |

The two attention kernels and the four GEMM kernels are each **one templated body**; upstream
shipped them as separate near-identical copies.

### Results — 8B shapes, M=2048, one layer, GPU timestamps

| stage | upstream on this machine | ported | speedup |
|---|---|---|---|
| QKV projections (x3) | 12.418 ms | **9.293** | 1.34x |
| causal attention (FP16 KV) | 17.24 ms | **1.573** | 11.0x |
| causal attention (Q8_0 KV) | 17.27 ms | **1.348** | 12.8x |
| output projection | 4.19 ms | **3.163** | 1.32x |
| MLP SwiGLU + down | 60.26 ms | **32.288** | 1.87x |
| **total / layer** | **94.15 ms** | **46.16** | **2.04x** |
| 32-layer prefill | 3013 ms | **1477 ms** | 2.04x |
| throughput | 21,752 tok/s | **44,364 tok/s** | |

Against the engine's own baseline kernels the ratio goes from 1.19x to **2.35x**. Against a
cold MLX reference for the same layer (44.43 ms) the engine moves from 47% to **96.3%**.

### The two findings worth carrying upstream

**1. The gate/up fusion is a net loss on a wide chip.** Simply *unfusing* the existing
`fused_gate_up_swiglu_q4_0` -- dispatching `pipe_gemm_q4_0_32x32` twice plus the existing
`swiglu_activation`, no new kernel code -- is worth **1.63x** (45.86 -> 28.15 ms at M=2048).
The fusion saves one memory pass worth ~0.4 ms here and pays `acc_g[32] + acc_u[32]` = 64
float accumulator registers per thread, worth ~17 ms.

**2. The Q8_0 KV cache never paid off before.** Upstream, Q8_0 attention cost 17.27 ms
against FP16's 17.24 ms -- halving the KV cache bought nothing, because the kernel was not
waiting on bytes. After the port it is 1.354 vs 1.576 ms, so the memory saving is finally
real and the feature earns its accuracy cost.

### Numerical fidelity improved

The scalar kernels sum partial products in `half` before promoting to `float`; the matrix
units accumulate in `float` throughout. Every residual got smaller:

| check | upstream | ported |
|---|---|---|
| engine gate, FP16 KV, M=512 | 0.00195 | **0.00098** |
| engine gate, Q8_0 KV, M=1024 | 0.02344 | **0.01562** |
| engine gate RMSE, M=128 | 0.00011 | **0.00007** |
| isolated attention vs CPU FP32, M=2048 | 0.02344 (published) | **0.00034** |
| isolated Q4_0 GEMM vs CPU FP32 | 0.00013-0.00085 | **0.00005-0.00035** |

Cosine similarity 1.000000, zero NaN/Inf, bit-identical across repeated runs, verified at
sequence lengths 1, 33, 63, 64, 65, 127, 128, 129, 255, 512, 1023, 1024, 2047, 2048, 4096, 8192.

### Building and running

```
make                      # no Xcode needed; shaders compile at runtime from source
./bench_8b_engine         # full 8B engine, ported kernels (default)
./attn_sg_bench           # isolated attention A/B + CPU FP32 reference
./mlp_bench               # isolated MLP A/B (fused/unfused, scalar/simdgroup) + CPU reference
```

Fall back to the upstream kernels at runtime, one axis at a time:

```
M3_SG_ATTN=0 ./bench_8b_engine     # upstream scalar attention
M3_SG_GEMM=0 ./bench_8b_engine     # upstream scalar GEMMs
M3_SG_ATTN=0 M3_SG_GEMM=0 ./bench_8b_engine   # fully upstream
```

### Limitations

* Only the **8B path** (`unified_8b_kernels.metal` + `bench_8b_engine`) is ported. The 1B
  path and the `micro_bench` / `pipelined_bench` / `flash_attn_bench` / `thermal_stress_test`
  harnesses still use `unified_kernels.metal` and are untouched.
* Diagnosis used achieved-throughput-vs-reference plus source analysis, **not** Xcode GPU
  counters -- the Metal Debugger needs full Xcode, and this machine has Command Line Tools
  only. Occupancy claims are inferred from register/threadgroup-memory arithmetic and
  confirmed by the resulting speedups, not read off a counter.
* The new kernels are **not double-buffered**; the upstream ones were. This began as untested
  headroom and became a reasoned rejection: double buffering doubles the staged tiles, and
  every tile measurement here says threadgroup memory is the binding constraint.
* Attention `BC` was swept: 32 is 1.3% slower for 44% more threadgroup memory, so 16 stays.
* MLP is still ~7% behind MLX, the largest remaining per-stage gap.
* Tile parameters were swept under a register-budget constraint (`BM`, `BK`, `BN`, simdgroups
  per threadgroup); the negative results are recorded in `M3_ULTRA_FINDINGS.md` alongside the
  wins, because two of them looked like free money on paper.

### Verification pass (corrections to the above)

Re-checked with two probes added in this branch, `occupancy_probe` and `roofline_probe`.
Full detail in `M3_ULTRA_FINDINGS.md`.

* **The bandwidth ceiling was mislabelled.** 694 GB/s came from an MLX read+write elementwise
  op. Measured directly: **776 GB/s read-only** (95% of 819.2 theoretical), 695 GB/s read+write
  -- which reproduces the MLX figure independently, so the number was right for what it
  measured and wrong as a reference for weight streaming, which is read-dominated. The
  conclusion (nothing is bandwidth-bound) is unaffected and slightly strengthened.
* **A register-spill claim is withdrawn.** The 64-row-tile negative result was attributed
  partly to register spilling. `occupancy_probe` reports `maxTotalThreadsPerThreadgroup = 1024`
  for all 16 kernels *including that one*, so the compiler applied no register-driven
  threadgroup limit anywhere. The 32 KB of threadgroup memory -- the full per-threadgroup
  budget, so one threadgroup per core -- explains it on its own.
* **No cross-die knee exists.** Coalesced read bandwidth climbs monotonically to 776 GB/s and
  is flat from 160 to 10,240 threadgroups (40K to 2.6M threads). Metal exposes no die-affinity
  control either, so there is nothing to tile against in principle.
* **Access pattern is worth up to 3.09x at matched thread counts** (784.9 vs 254.3 GB/s at 320
  threadgroups). `sgg_load_b` sits in the scattered regime: weights are `[n][k_block]`, so
  adjacent thread-pairs read q4_0 blocks `num_kb * 18` bytes apart (2,304 B at K=4096) and each
  18-byte block is read twice. Repacking to `[k_block][n]` at load time is the top remaining
  lead; not attempted here because it changes the weight layout.
* **GPU counters are unreachable on the test machine.** `MTLDevice.counterSets` exposes only
  `timestamp` -- no `stageutilization`, no `statistic`, and dispatch-boundary sampling is
  unsupported. Those need Instruments, which needs full Xcode. The occupancy reasoning here is
  inferred, not counter-verified, which is exactly how the withdrawn claim went wrong.
