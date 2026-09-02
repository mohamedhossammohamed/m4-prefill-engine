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
| QKV projections (x3) | 12.418 ms | **9.126** | 1.36x |
| causal attention (FP16 KV) | 17.24 ms | **1.583** | 10.9x |
| causal attention (Q8_0 KV) | 17.27 ms | **1.343** | 12.9x |
| output projection | 4.19 ms | **3.093** | 1.35x |
| MLP SwiGLU + down | 60.26 ms | **31.646** | 1.90x |
| **total / layer** | **94.15 ms** | **45.21** | **2.08x** |
| 32-layer prefill | 3013 ms | **1447 ms** | 2.08x |
| throughput | 21,752 tok/s | **45,302 tok/s** | |

Against the engine's own baseline kernels the ratio goes from 1.19x to **2.46x**. Against a
cold MLX reference for the same layer (44.43 ms) the engine moves from 47% to **98.3%**.

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
* MLP is still ~5% behind MLX, the largest remaining per-stage gap. QKV is now marginally
  ahead of MLX (9.126 vs 9.198 ms).
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

### Weight layout (added after the verification pass)

The q4_0 weight loader had two defects that are worth flagging because **they are not
M3-specific** -- the same gather is happening on M4:

1. Weights stored `[n][k_block]` mean adjacent thread-pairs read blocks `num_kb * 18` bytes
   apart (2,304 B at K=4096). `roofline_probe` measures that pattern as costing up to 3.09x at
   matched thread counts on this chip.
2. Every 18-byte block was read *twice*, because low nibbles of `qs[0..15]` are k 0-15 and high
   nibbles of the same bytes are k 16-31, so both threads of a pair needed all 16 bytes.

Fixed by a one-time repack to `[k_block][N]` at weight load, plus a byte-split so each thread
takes 8 distinct `qs` bytes and emits both nibble ranges. Adjacent threads now read adjacent
blocks and nothing is read twice. Worth 2.5% on the fused gate/up, 2.7% on the down projection,
2.0% on the whole layer. Both layouts stay resident so the baseline kernels are unaffected.

The gain is an order of magnitude below what the raw 3.09x pattern ratio suggests, because the
weight load overlaps compute rather than serialising against it. Recorded that way in
`M3_ULTRA_FINDINGS.md` so the microbenchmark is not read as a promise.
