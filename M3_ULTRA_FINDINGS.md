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
