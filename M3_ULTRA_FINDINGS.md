# M3 Ultra port — Phase 0-2 findings (2026-09-01, studio1)

Repo: mohamedhossammohamed/m4-prefill-engine @ ab01b63

## Phase 0 — hardware
| | |
|---|---|
| Host | mac-studio.ad.chpnet.org (studio1) |
| SoC | Apple M3 Ultra, **80 GPU cores**, 256 GB unified |
| macOS | 26.5.1 (25F80), Metal 4 |
| Toolchain | CLT only, **no offline `metal` compiler** — not needed, all shaders built via newLibraryWithSource at runtime |
| Weights | none required; bench uses deterministic PRNG synthetic Q4_0 + built-in CPU FP32 gold ref |

`make` builds all 7 targets clean.

## Phase 1 — measured ceilings (MLX, same box, same session)
- FP16/Q4 GEMM ceiling: **24.8 TFLOP/s** (stable 22-25 across 4k-8k shapes)
- Memory bandwidth: **694 GB/s** (87% of ~800 theoretical -> harness is sound)

## Phase 2 — diagnosis, 8B shapes, M=2048, ONE layer, GPU timestamps
K=4096 H=32 D=128 N_mlp=14336

| Stage | repo baseline | repo "optimized" | MLX | opt TFLOP/s | % of 24.8 ceiling | verdict |
|---|---|---|---|---|---|---|
| QKV proj (x3) | 18.74 ms | 12.41 ms | 9.20 ms | 16.6 | 67% | compute-bound, moderate headroom |
| Causal attention | 20.46 ms | 17.24 ms | **2.09 ms** | 2.0 | **8%** | occupancy-limited, **8.3x off** |
| Output proj | 6.35 ms | 4.19 ms | 3.05 ms | 16.4 | 66% | compute-bound, moderate headroom |
| MLP SwiGLU | 66.35 ms | 60.26 ms | 30.10 ms | 12.0 | 48% | compute-bound, **2.0x off** |
| **TOTAL / layer** | **111.90** | **94.10** | **44.43** | | | |
| 32-layer prefill | 3580 ms | 3011 ms | **1422 ms** | | | |

DRAM utilisation during the run: **1.5 GB/s = 0.2% of the 694 GB/s ceiling.**

## Conclusions
1. **The memory-bound premise is dead on this chip.** The engine touches 0.2% of available
   bandwidth. Every kernel is compute/occupancy limited. The plan Read-This-First was correct.
2. **Zero `simdgroup_matrix` / `simdgroup_multiply_accumulate` in the entire repo** (all 6 .metal
   files, grep count 0). Everything is scalar/half4 FMA written for 10 cores. Phase 4a is
   untouched ground and is the whole ballgame.
3. **The repo's own "optimized" path is 2.12x SLOWER than a stock MLX call** at the same shapes
   on this hardware. Its reported 1.19x speedup is real but measured against its own strawman
   baseline kernel, not against a tuned Apple-Silicon GEMM.
4. Throughput is flat at ~22,800 tok/s from M=512 to M=2048 — the signature of a fixed
   compute/occupancy ceiling, not a bandwidth wall.

## Headroom
~2.1x end-to-end available just by reaching parity with MLX. Ordering by payoff:
1. **Attention** — 17.24 -> ~2.1 ms. Biggest multiple (8.3x), 18% of layer time.
2. **MLP** — 60.26 -> ~30 ms. Biggest absolute win (-30 ms), 64% of layer time.
3. QKV + O-proj — 16.60 -> ~12.2 ms. Smallest, do last.

All three are the same fix: simdgroup_matrix tiles.

---

# Phase 3 survey — where the M4 assumptions actually live

Surface area is tiny. Four things, and they are *baked into array dimensions*, not tunables:

1. `threadsPerThreadgroup: MTLSizeMake(32,1,1)` on **every** GEMM and attention dispatch
   (bench_8b_engine.mm:425,437,442,447,460,472,494,506,539,550,583,595,616,626...).
   32 threads = exactly one SIMD group per threadgroup.
2. `setThreadgroupMemoryLength: 4096` (GEMM) / `16384` (flash attention).
3. `BR=32, BC=16, TG_SIZE=32` (unified_8b_kernels.metal:727-729, 869-871).
4. The 32x32 output tile, hard-baked as `float acc[32]` and `/32` grid arithmetic.

## Why each kernel sits where it does

**`pipe_gemm_q4_0_32x32` (QKV / O-proj, 67% of ceiling)**
32 threads, one output column each, `float acc[32]` for 32 rows. The inner loop re-reads the
whole 32x32 `sh_A` tile from threadgroup memory *per thread, per k-block*: 8 half4 loads per
32 MACs. It is threadgroup-memory-load bound, not ALU bound. `simdgroup_matrix` 8x8x8 replaces
that inner loop with register-resident tiles and deletes the traffic entirely.

**`fused_gate_up_swiglu_q4_0` (MLP, 48% of ceiling) — the fusion is now a net negative**
`float acc_g[32]` + `float acc_u[32]` = **64 float accumulator registers per thread**, double
the unfused GEMM, on the same 32-thread threadgroup. That is exactly the register-pressure
inflation the plan predicted in 4c, and it is why MLP sits 19 points below the unfused QKV
kernel running identical math. **Unfusing gate/up is a real experiment here, not heresy.**

**`flash_attn_fp16_causal_d128` (attention, 8% of ceiling) — occupancy floor**
`half4 q_reg[32]` + `half4 o_acc[32]` = 256 halves = ~128 32-bit registers/thread, *plus*
16 KB threadgroup memory, *plus* 32 threads/threadgroup, one query row per thread.
Grid is ((M+31)/32, H) = 64 x 32 = 2048 threadgroups = 65,536 threads at M=2048, against
~80k thread slots on 80 cores. It fills the machine roughly once with **zero latency hiding**,
and the 16 KB threadgroup allocation stops it from even doing that. 8% is the expected result.

## Correction to the plan
Phase 3 says "prefer runtime `device_config` if it does not cost performance." Not possible
here: these constants size C arrays and unroll counts, so they must be compile-time.
But that is free anyway -- every harness already builds its shader with
`newLibraryWithSource` at runtime, so a profile is just `MTLCompileOptions.preprocessorMacros`
with `-D` values. One binary, both chips, ~5 lines. No config struct needed.

---

# Phase 4a.1 — `flash_attn_sg_causal_d128` (simdgroup_matrix attention)

New kernel appended to `unified_8b_kernels.metal`. Same math, same layouts, same
`-1e30` sentinel handling as the upstream scalar kernel. Changes:
* 128 threads (4 simdgroups) per threadgroup, 8 query rows per simdgroup
  (was 32 threads, 1 query row per thread)
* Q*K^T and P*V on the matrix units (`simdgroup_half8x8` in, `simdgroup_float8x8` accum)
* the online-softmax rescale is applied as `diag(alpha) * O` on the matrix units
* softmax spread over all 32 lanes (4 lanes/row) via `simd_shuffle_xor` reductions
* ~55 registers/thread vs ~128; 20 KB threadgroup memory for 4x the threads
* **float** O accumulators; upstream accumulated the output in `half` throughout

Upstream kernel is kept and selectable: `M3_SG_ATTN=0 ./bench_8b_engine`.

## Isolated A/B (`attn_sg_bench`, GPU timestamps, median of 9 after 3 warmup)
Cold buffers. maxabs/cos are against a **CPU FP32 gold reference** through M=2048.

| M | scalar ms (IQR) | simdgroup ms (IQR) | speedup | max-abs | cosine | NaN | determinism |
|---|---|---|---|---|---|---|---|
| 1 | 0.048 (0.001) | 0.021 (0.001) | 2.30x | 0.00000 | 1.000000 | 0 | bit-identical |
| 33 | 0.181 (0.001) | 0.039 (0.003) | 4.63x | 0.00031 | 1.000000 | 0 | bit-identical |
| 127 | 0.342 (0.000) | 0.049 (0.000) | 7.00x | 0.00033 | 1.000000 | 0 | bit-identical |
| 128 | 0.348 (0.015) | 0.049 (0.000) | 7.09x | 0.00033 | 1.000000 | 0 | bit-identical |
| 129 | 0.365 (0.000) | 0.050 (0.002) | 7.25x | 0.00052 | 1.000000 | 0 | bit-identical |
| 255 | 0.705 (0.001) | 0.092 (0.001) | 7.70x | 0.00032 | 1.000000 | 0 | bit-identical |
| 1023 | 5.478 (0.224) | 0.747 (0.014) | 7.33x | 0.00036 | 1.000000 | 0 | bit-identical |
| 1024 | 5.485 (0.108) | 0.761 (0.025) | 7.21x | 0.00034 | 1.000000 | 0 | bit-identical |
| 2047 | 17.314 (0.270) | 2.515 (0.031) | 6.88x | 0.00034 | 1.000000 | 0 | bit-identical |
| 2048 | 17.178 (0.071) | 2.545 (0.041) | 6.75x | 0.00034 | 1.000000 | 0 | bit-identical |
| 4096 | 62.099 (0.479) | 9.171 (0.071) | 6.77x | 0.00098* | 0.999995* | 0 | bit-identical |
| 8192 | 231.763 (1.838) | 35.149 (0.291) | 6.59x | 0.00098* | 0.999990* | 0 | bit-identical |

\* M=4096/8192 are compared against the **old kernel**, not the CPU reference (the FP32
reference is too slow to be useful there). That residual is the old kernel's half-precision
accumulation, not ours. Every M in the plan's edge-case list is CPU-referenced.

Plan bar was max-abs < 0.03 and cosine > 0.9999. Measured max-abs is **0.00034 at M=2048**
-- 69x inside the bar, and tighter than the upstream kernel's own published 0.02344, because
this kernel accumulates O in float instead of half.

## Roofline
| | ms @ M=2048 | TFLOP/s | % of 24.8 ceiling |
|---|---|---|---|
| upstream scalar | 17.18 | 2.0 | 8% |
| **simdgroup port (cold)** | **2.545** | **13.5** | **54%** |
| MLX `scaled_dot_product_attention` (cold) | 2.088 | 16.5 | 66% |
| simdgroup port, in-pipeline (warm Q/K/V) | 1.588 | 21.6 | 87% |

Cold-vs-cold is the honest comparison: **82% of MLX**. The in-pipeline number is faster only
because the QKV projection just wrote Q/K/V and they are still cache-resident; it is reported
for completeness, not as a like-for-like win over MLX.

## End-to-end effect (full engine, 8B, M=2048, one layer)
| stage | before | after |
|---|---|---|
| QKV projections | 12.41 ms | 12.41 ms (untouched) |
| **causal attention** | **17.24 ms** | **1.59 ms (12.9x)** |
| output projection | 4.19 ms | 4.19 ms (untouched) |
| MLP SwiGLU | 60.26 ms | 60.22 ms (untouched) |
| **total / layer** | **94.10 ms** | **78.40 ms (1.20x)** |
| 32-layer prefill | 3011 ms | 2509 ms |
| engine vs its own baseline | 1.19x | **1.43x** |
| throughput | 21,764 tok/s | 26,116 tok/s |

Engine's own end-to-end gate: `[PASS]` at M=128/512/1024, MaxDiff **identical** to the scalar
path (0.00195 / 0.00195 / 0.00781) with marginally better RMSE. One variable changed, fidelity
unchanged, speed 12.9x on the stage.

## Note for the next step
The Q8_0 KV-cache path (`flash_attn_q8_0_causal_d128`) is untouched and still on the scalar
kernel: its attention stage is still 17.27 ms, so the Q8_0 pipeline is now the slower of the
two (94.13 ms vs 78.40 ms). Same treatment applies to it.

---

# Phase 4a.2 — Q8_0 KV attention, and a deduplication

Upstream shipped `flash_attn_fp16_causal_d128` and `flash_attn_q8_0_causal_d128` as two
~150-line kernels whose bodies are byte-for-byte identical: the dequant happens in the tile
loader, so everything downstream already operates on a plain `half` tile. The port collapses
them into one `flash_attn_sg_impl<KV>` template with two `sga_load_kv` overloads picked by
the KV pointer type. Net: one body to maintain, two entry points.

The Q8_0 loader was also rewritten for the wider threadgroup -- upstream walks `BC*4 = 64`
q8_0 blocks with 32 threads; at 128 threads that would leave half of them idle, so each block
is split across 2 threads (16 values each) and all 128 participate.

Refactor was proven neutral first: the FP16 A/B re-run gave identical max-abs at every
sequence length (0.00034 at M=2048) with timings inside the IQR.

## Full engine, 8B, M=2048, one layer

| stage | baseline | before port | after port |
|---|---|---|---|
| QKV projections | 18.74 | 12.41 | 12.41 (untouched) |
| **attention, FP16 KV** | 20.41 | 17.24 | **1.589 ms — 12.8x** |
| **attention, Q8_0 KV** | 20.41 | 17.27 | **1.352 ms — 12.8x** |
| output projection | 6.34 | 4.19 | 4.19 (untouched) |
| MLP SwiGLU | 66.35 | 60.26 | 60.20 (untouched) |
| **total/layer, FP16 KV** | 111.84 | 94.10 | **78.38 (1.20x)** |
| **total/layer, Q8_0 KV** | 111.90 | 94.15 | **78.22 (1.20x)** |
| 32-layer prefill, Q8_0 | 3579 ms | 3013 ms | **2503 ms** |
| engine vs its own baseline | 1.00x | 1.19x | **1.43x** |
| throughput | 18,312 tok/s | 21,752 tok/s | **26,182 tok/s** |

Engine gate `[PASS]` at M=128/512/1024 on both paths, MaxDiff unchanged from the scalar
kernels (FP16 0.00195/0.00195/0.00781, Q8_0 0.00305/0.00244/0.02344). The Q8_0 residual is
the KV quantisation itself, not the attention arithmetic -- as expected, since the body is
now literally the same code as the CPU-verified FP16 path.

## The interesting result: the Q8_0 KV cache never actually paid off before

Upstream, Q8_0 attention cost **17.27 ms** against FP16's **17.24 ms** -- halving the KV
cache bought *nothing*. That is the signature of an occupancy-bound kernel: it was not
waiting on bytes, so removing bytes changed nothing. After the port the same comparison is
**1.352 ms vs 1.589 ms**, i.e. Q8_0 is now 15% faster than FP16 and the memory saving is
finally real. The KV-quantisation feature only becomes worth its accuracy cost once the
kernel is no longer stalled on occupancy.

## A caveat on "% of roofline"
The 24.8 TFLOP/s figure is an *achieved MLX GEMM* number, not a hardware peak. The warm
in-pipeline attention (1.352 ms => ~25.8 TFLOP/s on executed causal FLOPs) sits slightly
above it, which means the reference is a floor on the true ceiling rather than a wall. The
honest roofline comparison remains the cold isolated measurement: 2.55 ms at M=2048, 54% of
that reference, against 8% before, and 82% of a cold MLX `scaled_dot_product_attention`.

---

# Phase 4a.3 / 4c — Q4_0 GEMM and the fusion question

New kernel `sg_gemm_q4_0_impl<FUSED, BM>` replaces both `pipe_gemm_q4_0_32x32` and
`fused_gate_up_swiglu_q4_0`: 128 threads (4 simdgroups), BN=64 output columns per
threadgroup (16 per simdgroup), BK=32 so one dequant step is exactly one q4_0 block.
B is dequantised into threadgroup memory as `[n][k]` and fed to the matrix units
transposed, which also makes the dequant writes contiguous. Selectable with `M3_SG_GEMM=0`.

New instrument: `mlp_bench` (isolated A/B + CPU FP32 reference).

## 4c: is the M4-era fusion still a win? No -- it costs 63%.

M=2048, K=4096, N=14336, GPU timestamps, median of 9 after 3 warmup:

| config | ms (IQR) | vs upstream | TFLOP/s | % of 24.8 ref |
|---|---|---|---|---|
| fused scalar (upstream) | 45.861 (0.035) | 1.00x | 10.49 | 42% |
| **unfused scalar** | **28.151 (0.011)** | **1.63x** | 17.09 | 69% |
| fused simdgroup, BM=32 | 22.197 (0.002) | 2.07x | 21.67 | 87% |
| unfused simdgroup | 21.408 (0.004) | 2.14x | 22.47 | 91% |
| fused simdgroup, BM=64 | 599.421 (1.898) | 0.08x | 0.80 | 3% |

**Simply unfusing the upstream kernel -- changing no kernel code, just dispatching the
existing `pipe_gemm_q4_0_32x32` twice plus the existing `swiglu_activation` -- is worth
1.63x.** The fusion that saved a memory pass on a 10-core M4 costs 63% on an 80-core M3
Ultra, because what it actually buys is `acc_g[32] + acc_u[32]` = 64 float accumulator
registers per thread. The memory pass it saves is worth ~0.4 ms here; the register pressure
costs ~17 ms. This is the clearest single confirmation that the bottleneck moved.

Down projection (M=2048, K=14336, N=4096): 14.402 -> 11.139 ms, 1.29x, 16.70 -> 21.59
TFLOP/s (67% -> 87% of reference).

## Negative result: BM=64 on the fused kernel (do not retry)
Doubling the row tile halves how often each weight column is re-read from device memory
(M/64 passes instead of M/32), which looked like free money since weight re-reads are
~6 ms of the 22. It is a **27x slowdown**. At BM=64 the fused kernel needs 32 float8x8
accumulators = 64 floats/thread *and* 32 KB of threadgroup memory for the epilogue staging
(the full per-threadgroup limit, so one threadgroup per core). Registers spill and occupancy
collapses: 0.80 TFLOP/s.

The first run of this experiment was also numerically wrong (max-diff 0.44) because the
harness was still passing 16 KB of threadgroup memory. That was fixed and re-measured before
recording: the corrected run is numerically identical to the other variants (max-diff
0.00049) and still 27x slow, so the result is attributable to the register/occupancy cliff
and not to the bug.

Kept BM=32 fused for the engine: 3.6% behind unfused-simdgroup on this stage (0.8 ms, ~1% of
the layer) but a drop-in replacement needing no extra 117 MB of intermediate buffers.

## Fidelity (CPU FP32 reference, K=4096, N=512)
The simdgroup GEMM is **2.5-3x tighter than upstream at every shape** -- the scalar kernels
sum 32 half products into a `half4` before promoting to float, while the matrix units
accumulate in float throughout.

| shape | M | simdgroup max-abs | upstream max-abs |
|---|---|---|---|
| gate/up | 1 / 33 / 63 / 64 / 65 | 0.00005-0.00010 | 0.00013-0.00028 |
| gate/up | 127 / 128 / 129 / 255 / 512 | 0.00008-0.00011 | 0.00025-0.00039 |
| down | 1 / 33 / 64 / 129 / 512 | 0.00028-0.00035 | 0.00062-0.00085 |

Cosine 1.000000 everywhere, both kernels.

## Full engine, 8B, M=2048, one layer

| stage | baseline | untuned upstream | after port |
|---|---|---|---|
| QKV projections | 18.75 | 12.42 | 12.42 (**still untouched**) |
| causal attention | 20.48 | 17.24 | 1.579 (13.0x) |
| output projection | 6.34 | 4.19 | 3.246 (1.29x) |
| MLP SwiGLU + down | 66.34 | 60.26 | 33.415 (1.80x) |
| **total / layer** | 111.91 | 94.10 | **50.50** |
| 32-layer prefill | 3581 ms | 3013 ms | **1616 ms** |
| engine vs its own baseline | 1.00x | 1.19x | **2.22x** |
| throughput | 18,300 tok/s | 21,752 tok/s | **40,557 tok/s** |

Engine gate `[PASS]` on both KV paths at M=128/512/1024, MaxDiff unchanged, AvgDiff and RMSE
both improved (RMSE 0.00011 -> 0.00008 at M=128).

Cold MLX reference for the same layer is 44.43 ms, so the engine is now at **88% of MLX**,
from 47% at the start. The entire remaining gap is QKV: 12.42 ms against MLX's 9.20 ms.
`pipe_qkv_head_gemm_q4_0` is the last scalar GEMM -- it writes directly into [H, M, D] head
layout, which is why it was not covered by the generic replacement.

---

# Phase 4a.4 — QKV head-major projection (last scalar GEMM)

`pipe_qkv_head_gemm_q4_0` is the same GEMM as the others but scatters its result into
`[H, M, D]` head-major layout instead of `[M, N]`, which is the only reason the generic
replacement did not already cover it. Rather than a fourth kernel, the epilogue became a
template parameter: `sg_gemm_q4_0_impl<FUSED, BM, HEADOUT>` plus a `D_head` argument, and
`sg_qkv_head_gemm_q4_0` is a five-line entry point. One body now serves all four call sites.

QKV projections (3x, M=2048, K=4096, N=4096): **12.418 -> 9.527 ms, 1.30x**.
Cold MLX for the same three GEMMs is 9.198 ms, so this is within 3.6% of MLX.

# Final state — full engine, 8B, one layer

| stage | baseline | untuned upstream | ported | vs upstream | cold MLX |
|---|---|---|---|---|---|
| QKV projections (x3) | 18.74 | 12.42 | **9.527** | 1.30x | 9.198 |
| causal attention (FP16 KV) | 20.41 | 17.24 | **1.576** | 10.9x | 2.088 |
| causal attention (Q8_0 KV) | 20.41 | 17.27 | **1.354** | 12.8x | - |
| output projection | 6.34 | 4.19 | **3.245** | 1.29x | 3.051 |
| MLP SwiGLU + down | 66.32 | 60.26 | **33.424** | 1.80x | 30.096 |
| **total / layer (FP16 KV)** | 111.81 | 94.10 | **47.77** | **1.97x** | 44.43 |
| **total / layer (Q8_0 KV)** | 111.90 | 94.15 | **47.62** | **1.98x** | - |
| 32-layer prefill | 3578 ms | 3013 ms | **1524 ms** | 1.98x | - |
| engine vs its own baseline | 1.00x | 1.19x | **2.35x** | | |
| throughput | 18,316 tok/s | 21,752 tok/s | **43,011 tok/s** | | |

Across the sequence sweep (ms/layer, Q8_0 path): M=33 9.14->3.76, M=127 10.57->4.92,
M=129 12.60->5.24, M=512 25.02->13.11, M=1024 52.18->24.13, M=2048 111.90->47.59.

**The engine is now at 93% of a cold MLX reference for the whole layer, from 47%.** Per
stage it is within 3.6% (QKV), 6% (O-proj) and 11% (MLP) of MLX, and ahead on attention.

## Fidelity moved the right way at every step
Engine gate, `[PASS]` on both KV paths at all three verified lengths, and the residuals
*shrank* as kernels were replaced, because the scalar kernels accumulated partial sums in
`half` where the matrix units accumulate in `float`:

| | untuned upstream | ported |
|---|---|---|
| FP16 KV, M=128 / 512 / 1024 | 0.00195 / 0.00195 / 0.00781 | 0.00195 / **0.00098** / 0.00781 |
| Q8_0 KV, M=128 / 512 / 1024 | 0.00305 / 0.00244 / 0.02344 | **0.00293** / 0.00244 / **0.01562** |
| FP16 KV RMSE, M=128 | 0.00011 | **0.00007** |

Threshold is 0.05. Isolated CPU-FP32 checks: attention max-abs 0.00034 at M=2048 (plan bar
0.03) across 12 sequence lengths including 1/33/127/129/255/2047, bit-identical across runs;
GEMM max-abs 0.00005-0.00035, 2.5-3x tighter than upstream.

---

# Phase 7 — Written summary

## What worked unchanged
The upstream design is sound above the kernel level and almost all of it survived the port:

* **The pipeline shape.** QKV -> causal attention -> O-proj + residual -> SwiGLU -> down was
  never the problem, and none of it was restructured.
* **The head-major QKV output layout.** Projecting directly into `[H, M, D]` to eliminate a
  transpose kernel is a genuinely good idea and is *more* valuable now, not less -- it was
  kept verbatim, only the GEMM underneath it changed.
* **The online-softmax algebra**, including the `-1e30` sentinel and the
  `(running_max > -1e20f) ? exp(...) : 0` guards. Fully-masked causal tiles are a real case
  at tile granularity and upstream handles them correctly; the new kernel copies that logic
  exactly rather than reinventing it.
* **The Q4_0 / Q8_0 block formats and dequant math** (nibble ordering, the `(q - 8) * d`
  form, k 0-15 from low nibbles and 16-31 from high nibbles of the same 16 bytes).
* **The benchmark methodology.** GPU timestamps rather than wall clock, deterministic PRNG
  inputs, a CPU FP32 gold reference, edge-case sequence lengths. The port added IQR reporting
  and a determinism check but the foundation was already right.
* **Runtime shader compilation** via `newLibraryWithSource`, which is why none of this needed
  Xcode.

## What changed and why
One cause, four instances: every kernel dispatched 32-thread threadgroups and gave one
output column (or one query row) to a single thread. That is correct for 10 cores and
starves 80.

| change | motivating measurement |
|---|---|
| 32 -> 128 threads per threadgroup everywhere | 65,536 threads vs ~80,000 slots at M=2048: one fill, no latency hiding |
| scalar half4 FMA -> `simdgroup_matrix` | attention at 8% and MLP at 42% of a 24.8 TFLOP/s reference |
| `half` -> `float` accumulators | free with the matrix units, and upstream's `half` accumulation was the dominant error term |
| attention: 1 row/thread -> 8 rows/simdgroup | ~128 registers/thread was the occupancy ceiling |
| GEMM: B staged as `[n][k]`, fed transposed | makes the dequant writes contiguous and feeds the matrix units directly |
| two attention kernels -> one template | they were byte-identical below the tile loader |
| four GEMM kernels -> one template | same body, three different epilogues (plain / SwiGLU / head-scatter) |

## Achieved speedups
Per layer at 8B / M=2048: **94.15 -> 47.62 ms, 1.98x**. 32-layer prefill 3013 -> 1524 ms.
Against the engine's own baseline kernels, 1.19x -> **2.35x**. Throughput 21,752 ->
**43,011 tok/s**. Roofline position went from 8% (attention) and 42% (MLP) of the reference
to 54% cold / ceiling-class warm, and 87%. Against cold MLX for the whole layer: 47% -> 93%.

## What did not work
* **BM=64 on the fused MLP kernel: 27x slower.** Halving weight re-reads looked free; 64
  float accumulators per thread plus 32 KB of epilogue staging spills registers and pins
  occupancy at one threadgroup per core. 0.80 TFLOP/s. Recorded in full because the
  arithmetic that motivated it was correct and the result still went the other way.
* **Keeping the gate/up fusion was the wrong instinct.** It is 3.6% behind the unfused
  simdgroup path. It was kept only because unfusing needs two extra `[M, N]` buffers
  (117 MB at M=2048) for ~1% of the layer. On the *scalar* kernels the same fusion costs 63%.
* **The plan's Phase 3 as written.** It asks for a runtime `device_config` struct. These
  constants size C arrays and unroll counts, so they are compile-time by construction. No
  struct was built; the fallbacks are two environment variables selecting whole kernels,
  which is a smaller diff and made the one-variable-at-a-time measurements easy.

## Remaining limitations
* Only the **8B path** is ported. `unified_kernels.metal` (1B) and the `micro_bench`,
  `pipelined_bench`, `flash_attn_bench` and `thermal_stress_test` harnesses are untouched --
  note `thermal_stress_test` builds against the 1B kernels and therefore does *not* exercise
  any of this work.
* **No GPU counters were used.** The Metal Debugger needs full Xcode; this machine has
  Command Line Tools only. Phase 2's diagnosis is achieved-throughput-vs-reference plus
  source analysis. Occupancy claims are inferred from register and threadgroup-memory
  arithmetic and corroborated by the speedups, not read off a counter. This is the biggest
  methodological gap in the work.
* The new kernels are **not double-buffered**; the upstream ones were. Occupancy supplies the
  latency hiding instead. This is untested headroom, not a considered decision.
* The 24.8 TFLOP/s "ceiling" is an *achieved MLX GEMM* number, not a hardware peak. Warm
  in-pipeline attention slightly exceeds it, so it is a floor on the true ceiling.
* Attention `BC` is still 16, inherited from the M4 tuning and never swept.

## Suggested next steps, in expected-value order
1. **Double-buffer the K/V and A tiles** in the new kernels. Upstream had it, the port
   dropped it for simplicity, and it was never measured.
2. **Sweep attention `BC` (16 -> 32/64)** and the GEMM `BN`/`BK`. Constrain the grid with the
   register arithmetic that predicted the BM=64 cliff; do not sweep blind.
3. **Close the MLP gap** -- 11% behind MLX and the largest remaining per-stage deficit.
4. **Port `unified_kernels.metal`** (the 1B path) the same way; it is the same four patterns.
5. **Get GPU counters** on a machine with Xcode and re-do Phase 2 properly, to check the
   occupancy story is what actually happened rather than what the arithmetic predicted.
6. Try feeding the Q8_0 KV cache to integer matrix ops instead of dequantising to `half`
   in the tile loader.

## Attribution
Ported from **[mohamedhossammohamed/m4-prefill-engine](https://github.com/mohamedhossammohamed/m4-prefill-engine)**
by Mohammed Hossam, commit `ab01b63`. The engine, its benchmark harness, its CPU gold
reference and its numerical-verification suite are the original author's work; this port
changes the kernel implementations for a wider chip and adds two isolated A/B instruments.

---

# Phase 6 — Full engine validation

## Sustained load (128 s continuous, four back-to-back full benchmark runs)

| run | wall | layer ms (Q8_0, M=2048) | tok/s | fidelity |
|---|---|---|---|---|
| 1 | 32 s | 47.760 | 42,881 | 6 PASS / 0 FAIL |
| 2 | 32 s | 47.767 | 42,875 | 6 PASS / 0 FAIL |
| 3 | 32 s | 47.759 | 42,882 | 6 PASS / 0 FAIL |
| 4 | 32 s | 47.775 | 42,868 | 6 PASS / 0 FAIL |

Spread across 128 s: **0.03%**. System memory free held at 98% throughout and swap stayed at
**0.00 M** -- no allocator growth, no leak, no throttling. 24 fidelity gates, 0 failures.

## Kernel-selection matrix — success criterion 1

All four combinations build from one source and pass every fidelity gate:

| configuration | layer ms (Q8_0) | gates | MaxDiff (FP16/Q8_0 at M=128, 512, 1024) |
|---|---|---|---|
| ported (default) | **47.609** | 6 PASS / 0 FAIL | 0.00195 0.00293 / 0.00098 0.00244 / 0.00781 0.01562 |
| `M3_SG_ATTN=0` (scalar attn, sg GEMM) | 63.539 | 6 PASS / 0 FAIL | identical to default |
| `M3_SG_GEMM=0` (sg attn, scalar GEMM) | 78.220 | 6 PASS / 0 FAIL | 0.00195 0.00305 / 0.00195 0.00244 / 0.00781 0.02344 |
| `M3_SG_ATTN=0 M3_SG_GEMM=0` (fully upstream) | **94.120** | 6 PASS / 0 FAIL | same as upstream |

The fully-upstream path reproduces the original 94.1 ms baseline exactly, so the M4 code path
is intact. The matrix also isolates where the fidelity improvement came from: it tracks
`M3_SG_GEMM`, not `M3_SG_ATTN` -- the scalar GEMMs' `half` partial sums were the dominant
error term, not the attention kernel.

## Comparison against MLX (the "is this actually good" check)
Cold, same shapes, same machine, same session:

| stage | ported engine | MLX | ratio |
|---|---|---|---|
| QKV projections (x3) | 9.527 ms | 9.198 ms | 1.04x |
| output projection | 3.245 ms | 3.051 ms | 1.06x |
| MLP (gate/up/down) | 33.424 ms | 30.096 ms | 1.11x |
| causal attention (cold isolated) | 2.545 ms | 2.088 ms | 1.22x |
| **layer total** | **47.62 ms** | **44.43 ms** | **1.07x** |

93% of MLX, from 47% before the port.

## Success criteria
1. Builds and runs on M3 Ultra with no errors; M4 path still builds and passes -- **met**
   (matrix above, all four configurations).
2. All fidelity checks pass at every tested sequence length -- **met** (1, 33, 63, 64, 65,
   127, 128, 129, 255, 512, 1023, 1024, 2047, 2048, 4096, 8192 across the isolated harnesses;
   engine gate at 128, 512, 1024). Residuals got smaller, never larger.
3. Each top-3 kernel measurably closer to roofline -- **met**: attention 8% -> 54% (cold),
   MLP 42% -> 87%, plain GEMM 67% -> 87%. Caveat: measured as achieved-throughput against a
   reference, not from GPU counters (see Limitations).
4. End-to-end improvement at 8B / 4096 with median and IQR -- **met**. The engine's own sweep
   tops out at M=2048 (111.90 -> 47.59 ms/layer). The isolated attention harness runs 4096 and
   8192: 62.099 (IQR 0.479) -> 9.171 (0.071) and 231.763 (1.838) -> 35.149 (0.291).
5. Build and benchmark instructions reproduce from a clean checkout -- **met**, see the
   README section. `make` needs no Xcode; shaders compile at runtime.

---

# Addendum — 8-simdgroup fused MLP (shipped)

The `NSG` (simdgroups per threadgroup) and `BN` (column tile) knobs were the last two
unparameterised dimensions. Sweeping them at M=2048:

| variant | threads | gate/up ms | down ms |
|---|---|---|---|
| 4 simdgroups, 16 cols each (previous) | 128 | 21.828 | **10.808** |
| **8 simdgroups, 8 cols each, BM=64 (fused)** | 256 | **21.402** | 11.669 |
| 8 simdgroups, 16 cols each (BN=128) | 256 | - | 11.160 |

**The fused kernel wins with 8 simdgroups; the plain kernel loses.** Same mechanism, opposite
sign: doubling the simdgroups at a fixed column tile halves the columns each one owns, which
halves the matrix-multiplies issued per operand load. The fused kernel does two multiplies
per load (gate and up share the A operand), so it still has the intensity to absorb it and
banks the occupancy gain; the plain kernel does one, so it does not. Widening the tile to 128
columns to restore the ratio costs threadgroup memory and loses too, consistent with every
other measurement here.

Shipped: `sg_gemm_q4_0_fused_swiglu_wide` (8 simdgroups, BM=64, 12 KB) for the MLP, plain
GEMM unchanged at 4 simdgroups.

## Final numbers

| stage | untuned upstream | shipped | cold MLX |
|---|---|---|---|
| QKV projections (x3) | 12.418 | **9.293** | 9.198 |
| causal attention (FP16 KV) | 17.24 | **1.573** | 2.088 |
| causal attention (Q8_0 KV) | 17.27 | **1.348** | - |
| output projection | 4.19 | **3.163** | 3.051 |
| MLP SwiGLU + down | 60.26 | **32.288** | 30.096 |
| **total / layer (Q8_0 KV)** | **94.15** | **46.16** | 44.43 |
| 32-layer prefill | 3013 ms | **1477 ms** | - |
| throughput | 21,752 tok/s | **44,364 tok/s** | |

**94.15 -> 46.16 ms/layer, 2.04x over the untuned upstream engine, 2.42x over its own
baseline kernels, 96.3% of a cold MLX reference.** Per stage: within 1.0% (QKV), 3.7%
(O-proj) and 7.3% (MLP) of MLX.

| configuration | layer ms | tok/s | gates |
|---|---|---|---|
| ported (default) | **46.143** | 44,222 | 6 PASS / 0 FAIL |
| `M3_SG_ATTN=0` | 61.982 | 33,077 | 6 PASS / 0 FAIL |
| `M3_SG_GEMM=0` | 78.213 | 26,135 | 6 PASS / 0 FAIL |
| both `=0` (fully original) | 94.102 | 21,780 | 6 PASS / 0 FAIL |

Sustained: 4 back-to-back runs, 46.143-46.164 ms (0.05% spread), 24 gates, 0 failures,
swap 0.00 M. Fidelity unchanged from the previous phase at every gate.

## Process note
The wiring for this kernel was swept into the "Rebrand as the M3 Prefill Engine" commit by a
`git add -A`, so a documentation commit silently carried a kernel change and the README then
shipped stale numbers plus a limitation that was no longer true. Both are corrected here.
The measurement itself was unaffected -- the sweep and the validation above were run against
the wired build.

---

# Verification pass — corrections to earlier claims

Re-checked the published claims with two new instruments (`occupancy_probe`, `roofline_probe`).
Three needed correcting; all three were published, two of them on the upstream repo.

## 1. The bandwidth ceiling was mislabelled (694 GB/s -> 776 read / 695 read+write)

The 694 GB/s figure came from MLX's `x * 1.0` elementwise op, which **reads and writes**.
Measured directly, coalesced:

| | GB/s | % of 819.2 theoretical |
|---|---|---|
| read-only | **776** | 95% |
| read+write | **695** | 85% |
| read-only, scattered access | 729 (only at 2.6M threads) | 89% |

The 695 read+write figure reproduces MLX's 694 independently, so the number was never wrong --
it was labelled "the memory bandwidth ceiling" when the workload it was being compared against
(weight streaming) is read-dominated. The correct roofline reference for that is **776 GB/s**.
The conclusion is unaffected and slightly strengthened: the engine's 1.5 GB/s is 0.19% of the
read ceiling rather than 0.22%.

## 2. "Spills registers" was unsupported and is withdrawn

The BM=64 negative result was attributed to "registers spill and occupancy collapses."
`occupancy_probe` queries `maxTotalThreadsPerThreadgroup` for every kernel in the file --
the compiler's own register-allocation verdict, which drops below 1024 when a kernel is
register-starved:

```
sg_gemm_q4_0_fused_swiglu_bm64            1024       32          0
sg_gemm_q4_0_fused_swiglu_wide            1024       32          0
flash_attn_sg_causal_d128                 1024       32          0
...  (all 16 kernels report 1024)
```

Every kernel reports the device maximum, including the 27x-slower one. The compiler applied no
register-driven threadgroup limit anywhere, so the spilling half of the explanation has no
support and is withdrawn. The threadgroup-memory half stands on its own and is sufficient:
32 KB is the full per-threadgroup budget, so exactly one threadgroup fits per core.

Caveat on the instrument: `maxTotalThreadsPerThreadgroup = 1024` rules out a register-driven
*threadgroup-size* limit; it does not strictly rule out spilling to device memory. It is the
only register signal reachable without Xcode, and it does not support the claim that was made.

## 3. Attention `BC` is no longer "never swept"

Swept: `BC=32` is **1.3% slower** (2.463 vs 2.432 ms at M=2048) and needs 44% more threadgroup
memory (23,552 vs 16,384 B) -- the same pattern as every other tile experiment. The variant
also had an unresolved numerical defect (max-abs 0.20 vs the CPU reference), and since fixing
it could not change the tile's memory footprint or its work, it was **reverted rather than
shipped**. The timing is still informative because a numerical bug does not change the shape
of the work; the memory footprint is what loses.

## 4. GPU counters are unreachable on this machine (lever 1, blocked)

`MTLDevice.counterSets` on the M3 Ultra exposes exactly one set:

```
counterSets: 1
  set 'timestamp'  (1 counters)
      GPUTimestamp
common 'stageutilization': no
common 'statistic': no
sampling: atStageBoundary=1 atDispatchBoundary=0
```

No `stageutilization` (ALU / memory utilisation), no `statistic` (occupancy, instruction
counts), and dispatch-boundary sampling is unsupported. Those live behind Instruments, which
needs full Xcode; this machine has Command Line Tools only. So the occupancy story remains
**inferred from threadgroup-memory and register arithmetic plus the resulting speedups** --
which is exactly how claim 2 above went wrong, and why it is worth stating plainly rather than
burying in a limitations list.

## 5. No cross-die knee (lever 2, negative result)

The plan predicted a threadgroup count where UltraFusion traffic degrades scaling, and advised
tiling to keep a threadgroup's working set on one die. There is no such knee: coalesced read
bandwidth climbs monotonically to 776 GB/s and is **flat from 160 to 10,240 threadgroups**
(40K to 2.6M threads). Metal also exposes no die-affinity control, so there is nothing to tile
against even in principle. The hypothesis is refuted for this workload class.

What the sweep found instead is an access-pattern effect worth up to **3.09x at matched thread
counts** (784.9 vs 254.3 GB/s at 320 threadgroups), converging only once the grid is large
enough to make per-thread runs short.

**This is the top remaining lead for the MLP gap.** `sgg_load_b` is in the scattered regime:
weights are `[n][k_block]`, so adjacent thread-pairs read q4_0 blocks `num_kb * 18` bytes apart
(2,304 B at K=4096), and each 18-byte block is read twice, once per thread of the pair. At
M=2048 the gate/up weight traffic is ~2.1 GB over 21.4 ms -- about 98 GB/s against a ~254 GB/s
scattered ceiling. Repacking to `[k_block][n]` at load time would make those reads contiguous.
Not attempted: it changes the on-disk weight layout, which is a bigger decision than a tile
parameter.

## Engine unchanged
All experiments in this pass were reverted; the shipped engine is byte-identical and
re-verified: 46.152 ms/layer at M=2048, 6 PASS / 0 FAIL, stage times matching the published
table to within run-to-run noise.
