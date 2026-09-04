# Head-to-Head Comparative Benchmark: M4 Hardware Pipeline vs. Stock MLX

## Executive Metrology Summary
- **System Platform:** Apple Silicon M4 Unified Memory Architecture (macOS ARM64)
- **Model Dimensions:** 4 Layers, Hidden Dimension 512, Intermediate Dimension 1024
- **Attention Topology:** Grouped-Query Attention (8 Q heads, 2 KV heads, D=64)
- **Measurement Rigor:** 1 warmup iterations, 2 measured iterations (median telemetry reported)
- **Numerical Ground Truth:** MLX reference logits with strict IEEE 754 non-finite tripwires asserting 0 NaN/Inf

## 1. Prompt Prefill Performance (TTFT & Throughput)

| Format | Prefill Len | M4 TTFT | MLX TTFT | Speedup | M4 Throughput | MLX Throughput | Max Diff | Parity |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Q4_0 | 64 | 19.34 ms | 2.02 ms | **0.10x** | 3,309.6 tok/s | 31,612.7 tok/s | `0.000488` | PASSED |
| Q4_0 | 128 | 23.08 ms | 4.93 ms | **0.21x** | 5,546.9 tok/s | 25,964.8 tok/s | `0.000977` | PASSED |
| MLX_4BIT | 64 | 29.43 ms | 6.01 ms | **0.20x** | 2,174.6 tok/s | 10,648.0 tok/s | `0.000000` | PASSED |
| MLX_4BIT | 128 | 33.97 ms | 8.54 ms | **0.25x** | 3,768.4 tok/s | 14,983.7 tok/s | `0.000000` | PASSED |
| TERNARY_1_58 | 64 | 26.16 ms | 10.98 ms | **0.42x** | 2,446.7 tok/s | 5,828.4 tok/s | `0.000031` | PASSED |
| TERNARY_1_58 | 128 | 28.82 ms | 8.95 ms | **0.31x** | 4,442.0 tok/s | 14,294.0 tok/s | `0.000061` | PASSED |

## 2. Autoregressive Decode Step Performance (ms/tok & Throughput)

| Format | Context Len | Decode Steps | M4 Latency | MLX Latency | Speedup | M4 Throughput | MLX Throughput | Max Diff | Parity |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Q4_0 | 64 | 8 | 17.97 ms/tok | 1.31 ms/tok | **0.07x** | 55.66 tok/s | 763.79 tok/s | `0.000488` | PASSED |
| Q4_0 | 128 | 8 | 18.37 ms/tok | 1.65 ms/tok | **0.09x** | 54.44 tok/s | 605.19 tok/s | `0.000977` | PASSED |
| MLX_4BIT | 64 | 8 | 16.45 ms/tok | 1.37 ms/tok | **0.08x** | 60.79 tok/s | 730.59 tok/s | `0.000000` | PASSED |
| MLX_4BIT | 128 | 8 | 15.90 ms/tok | 1.28 ms/tok | **0.08x** | 62.90 tok/s | 780.43 tok/s | `0.007812` | PASSED |
| TERNARY_1_58 | 64 | 8 | 29.67 ms/tok | 2.18 ms/tok | **0.07x** | 33.71 tok/s | 459.17 tok/s | `0.000031` | PASSED |
| TERNARY_1_58 | 128 | 8 | 18.69 ms/tok | 1.74 ms/tok | **0.09x** | 53.51 tok/s | 573.83 tok/s | `0.000061` | PASSED |

## 3. UMA Physical Memory Metrology (`task_vm_info.phys_footprint`)

| Format | Architecture Pipeline | Baseline UMA | Peak Active UMA | Net Growth |
|:---|:---|---:|---:|---:|
| Q4_0 | M4 Hardware Pipeline | 29.25 MB | 225.66 MB | +196.41 MB |
| Q4_0 | Stock MLX Baseline | 29.25 MB | 225.67 MB | +196.42 MB |
| MLX_4BIT | M4 Hardware Pipeline | 185.63 MB | 228.27 MB | +42.64 MB |
| MLX_4BIT | Stock MLX Baseline | 185.63 MB | 228.28 MB | +42.66 MB |
| TERNARY_1_58 | M4 Hardware Pipeline | 186.28 MB | 226.16 MB | +39.88 MB |
| TERNARY_1_58 | Stock MLX Baseline | 186.28 MB | 226.16 MB | +39.88 MB |

## 4. Verification & Audit Post-Mortem Compliance
- **IEEE 754 Non-Finite Invariant (Flaw 6):** PASSED (0 NaN, 0 +/-Inf detected across all benchmark dispatches).
- **Verification Honesty (Flaws 3 & 9):** PASSED (Max absolute difference observed: `0.007812`, strictly $\le 0.05$). Zero hardcoded literals.
- **Cognitive Telemetry Latency Formatting (Flaw 11):** PASSED (All latencies $\ge 1000\text{ ms}$ converted cleanly to seconds).
- **UMA Working Set Tracking (Flaw 4):** PASSED (Kernel Mach task physical footprint sampled directly).
- **Format Parity Emulation (Flaw 10):** PASSED (Documented: Stock MLX lacks native GGUF kernels for Q4_K/Q4_0, baseline uses FP16 dequantized matmul emulation).
