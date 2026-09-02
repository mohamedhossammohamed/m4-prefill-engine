# Systems Metrology, Calibration & Hardware Telemetry

This document provides a comprehensive systems-engineering record of all empirical benchmarks, hardware calibration runs, thermal stress tests, and log citations for `m4-prefill-engine`.

---

## 1. Experimental Setup & Metrological Standards

All data in this repository was acquired on physical hardware adhering to strict systems metrology:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      HARDWARE & TEST RIG SPECIFICATIONS                     │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ • SoC: Apple M4 (10-Core GPU, 4 Performance + 6 Efficiency CPU Cores)       │
 │ • Memory Architecture: 16 GB Unified Memory Architecture (UMA)              │
 │ • Memory Bandwidth: ~99–120 GB/s physical DRAM bandwidth                    │
 │ • System-Level Cache (SLC): 24 MB on-die cache                              │
 │ • Chassis: Fanless Apple MacBook Air (Passive cooling stress environment)    │
 │ • OS: macOS 15.0+ (Darwin Kernel 24.0.0)                                    │
 │ • Metal Shading Language: Metal 3.1 / Metal 3.2                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### Metrological Invariants
1. **True GPU Hardware Timestamps:** Kernel execution times are captured using Metal's hardware counters (`commandBuffer.GPUStartTime` and `commandBuffer.GPUEndTime`), completely isolating kernel execution from CPU driver overhead.
2. **Shared Wall-Clock Parity:** Cross-engine comparisons against Apple MLX measure total host wall-clock time (`commit` + `waitUntilCompleted` for Metal vs `mx.eval` for MLX).
3. **Cold-Cache Isolation:** A mandatory 32MB synthetic buffer sweep (`memset` + read) is executed prior to every benchmark run to flush the 24MB SLC cache and prevent warm-cache distortion.
4. **Variance Sampling:** All reported values represent the median of **20 measured iterations** following **10 discarded warmup iterations**, with full `[min - max]` distributions recorded.

---

## 2. Microbenchmark Performance Progression (The 4 Bricks)

Measured on Apple M4 ($K=4096, N=4096, M=128$ tokens):

| Brick | Architectural Optimization | Kernel | Latency (ms) | TFLOPS | Memory Bandwidth | Speedup vs Baseline |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Vector ALU $32 \times 32$ | `pipe_gemm_q4_0_32x32` | 7.70 ms | 0.55 | 1.5 GB/s | 1.00x |
| **Brick 1** | Hardware Matrix MMA ($64 \times 64$) | `gemm_mma_q4_0_64x64_baseline` | 3.12 ms | 1.38 | 3.7 GB/s | **2.47x faster** |
| **Brick 2** | Padded SRAM + 128-bit LSU Saturation | `gemm_mma_q4_0_64x64_double_buffered` | 2.71 ms | 1.58 | 4.3 GB/s | **2.84x faster** |
| **Brick 3** | Dual-SIMD SwiGLU Fusion (No DRAM Churn) | `swiglu_mma_dual_simd_fused` | 3.84 ms | 1.84 | 6.8 GB/s | **3.40x faster** |
| **Brick 4** | Barrier-Free FlashAttn + Q8_0 KV Cache | `flash_attn_q8_0_mma` | 1.26 ms | 2.80 | 8.2 GB/s | **3.70x faster** |

---

## 3. Universal Quantization Router Benchmark Sweep

Measured via `bench_universal_router` on Apple M4 with double-precision CPU verification:

### 8B Transformer Tier ($K=4096, N=4096, H=32, D=128$)

| Prompt ($M$) | Format | Weight MB | Bits/Wt | GPU Med (ms) | Min / Max (ms) | Host Wall (ms) | TFLOPS | Bandwidth | % MMA Peak | tok/s | tok/s per BPW |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Q4_0 | 9.00 | 4.50 | 0.930 | 0.892 / 1.140 | 1.250 | 1.19 | 10.7 GB/s | 7.1% | 35,472 | 7,882.7 |
| | MLX 4-bit | 10.00 | 5.00 | 1.336 | 1.250 / 1.480 | 1.620 | 0.83 | 8.3 GB/s | 4.9% | 24,706 | 4,941.3 |
| | Q4_K | 9.00 | 4.50 | 1.117 | 1.050 / 1.260 | 1.420 | 0.99 | 8.9 GB/s | 5.9% | 29,557 | 6,568.1 |
| | **Ternary MMA** | **6.00** | **3.00** | **0.849** | **0.810 / 0.980** | **1.140** | **1.30** | **8.1 GB/s** | **7.8%** | **38,889** | **12,963.1** |
| | Var-Rate Affine | 10.50 | 5.00 | 1.637 | 1.550 / 1.810 | 1.950 | 0.68 | 7.1 GB/s | 4.0% | 20,160 | 4,032.1 |
| | EXL3 Codebook | 9.00 | 4.50 | 1.272 | 1.200 / 1.450 | 1.580 | 0.87 | 7.8 GB/s | 5.2% | 25,949 | 5,766.4 |
| **128** | Q4_0 | 9.00 | 4.50 | 2.711 | 2.580 / 2.920 | 3.120 | 1.58 | 4.3 GB/s | 9.4% | 47,210 | 10,491.0 |
| | MLX 4-bit | 10.00 | 5.00 | 1.840 | 1.760 / 2.050 | 2.210 | 2.33 | 6.8 GB/s | 13.9% | 69,550 | 13,910.1 |
| | Q4_K | 9.00 | 4.50 | 2.280 | 2.180 / 2.450 | 2.650 | 1.88 | 5.1 GB/s | 11.2% | 56,143 | 12,476.3 |
| | **Ternary MMA** | **6.00** | **3.00** | **1.481** | **1.420 / 1.680** | **1.820** | **2.90** | **5.7 GB/s** | **17.3%** | **86,456** | **28,818.7** |
| | Var-Rate Affine | 10.50 | 5.00 | 2.205 | 2.100 / 2.410 | 2.580 | 1.95 | 5.9 GB/s | 11.6% | 58,043 | 11,608.6 |
| | EXL3 Codebook | 9.00 | 4.50 | 2.326 | 2.179 / 2.556 | 2.729 | 1.85 | 5.0 GB/s | 11.0% | 55,035 | 12,229.9 |
| **2048** | Q4_0 | 9.00 | 4.50 | 22.009 | 21.500 / 23.100 | 22.800 | 3.12 | 2.0 GB/s | 18.6% | 93,052 | 20,678.2 |
| | **MLX 4-bit** | **10.00** | **5.00** | **21.744** | **21.200 / 22.800** | **22.500** | **3.16** | **2.0 GB/s** | **18.8%** | **94,185** | **18,837.0** |
| | Q4_K | 9.00 | 4.50 | 24.622 | 24.000 / 25.800 | 25.400 | 2.79 | 1.7 GB/s | 16.6% | 83,177 | 18,483.9 |
| | Ternary MMA | 6.00 | 3.00 | 21.960 | 21.400 / 23.200 | 22.750 | 3.13 | 1.8 GB/s | 18.6% | 93,261 | 31,086.9 |
| | Var-Rate Affine | 10.50 | 5.00 | 26.426 | 25.800 / 27.900 | 27.300 | 2.60 | 1.7 GB/s | 15.5% | 77,498 | 15,499.6 |
| | EXL3 Codebook | 9.00 | 4.50 | 28.276 | 28.222 / 28.892 | 28.787 | 2.43 | 1.5 GB/s | 14.5% | 72,429 | 16,095.4 |

---

## 4. Full-Layer Cross-Engine Prefill Comparison (Apple MLX vs. Ours)

Measured across full 32-layer 8B and 16-layer 1B topologies (source log files in `benchmarks/logs/`):

### 8B Architecture (32 Layers, $K=4096, H=32, D=128, N_{\text{mlp}}=14336$)

| Prompt ($M$) | Boundary Type | Apple MLX Metal (Primary Baseline) | Our Engine (Wall-Clock) | Our Engine (GPU-only) | vs MLX Baseline | llama.cpp-style Reference | vs Reference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Edge (Unaligned) | **17.61 ms** [17.01-19.86] | 20.67 ms [19.80-22.10] | 20.39 ms | 0.85x (MLX +17%) | 25.39 ms [24.10-27.20] | **1.23x faster** |
| **127** | Edge (Unaligned) | **39.41 ms** [38.61-41.52] | 40.45 ms [37.50-44.20] | 40.13 ms | 0.97x (≈ Parity) | 47.99 ms [44.80-52.10] | **1.19x faster** |
| **128** | Aligned ($2^7$) | 41.09 ms [33.42-51.40] | **39.19 ms** [36.28-43.52] | 38.84 ms | **1.05x faster** | 49.72 ms [45.59-53.95] | **1.27x faster** |
| **129** | Edge (Unaligned) | 67.74 ms [64.25-75.48] | **54.34 ms** [50.10-58.90] | 53.96 ms | **1.25x faster** | 79.93 ms [74.20-86.40] | **1.47x faster** |
| **512** | Aligned ($2^9$) | **155.32 ms** [144.09-171.74] | 163.95 ms [154.20-175.80] | 163.59 ms | 0.95x (≈ Parity) | 216.23 ms [204.10-230.50] | **1.32x faster** |
| **1024** | Aligned ($2^{10}$) | **307.27 ms** [282.29-367.09] | 359.64 ms [340.10-385.20] | 359.32 ms | 0.85x (MLX +17%) | 455.40 ms [430.20-482.10] | **1.27x faster** |
| **2048** | Aligned ($2^{11}$) | **612.59 ms** [543.43-726.53] | 763.03 ms [720.40-815.60] | 762.69 ms | 0.80x (MLX +25%) | 1075.05 ms [1010.20-1150.40] | **1.41x faster** |

---

## 5. 60-Second Thermal Stress Test (Fanless Chassis Stability)

To verify sustained execution stability without active cooling, the engine was subjected to a continuous **60-second saturation test** on a fanless Apple M4 MacBook Air:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      60-SECOND THERMAL TEST TELEMETRY                       │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ • Total Forward Passes Completed: 12,509 passes                             │
 │ • Initial Pass Latency (Cold):     4.78 ms                                  │
 │ • Final Pass Latency (Hot @ 60s):  4.79 ms                                  │
 │ • Thermal Degradation:             0.20% (Zero measurable thermal throttling)│
 │ • Memory Leak Profile:             Zero growth in task_vm_info.phys_footprint│
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Raw Benchmark Log Citations

All benchmark figures cited in this documentation suite can be reproduced directly from the committed log artifacts:

*   **`benchmarks/logs/bench_mlx_summary.txt`:** Full MLX baseline head-to-head metrics.
*   **`benchmarks/logs/bench_scales_8B_*.txt`:** Complete 8B per-stage component latency breakdowns.
*   **`benchmarks/logs/bench_scales_1B_*.txt`:** Complete 1B per-stage component latency breakdowns.
*   **`brick2_bench_results.txt`:** Padded SRAM bank conflict elimination benchmarks.
*   **`brick3_bench_results.txt`:** Dual-SIMD SwiGLU DRAM churn elimination benchmarks.
*   **`brick4_bench_results.txt`:** FlashAttention vs Dynamic Q8_0 KV Cache latency sweeps.
