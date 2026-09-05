# Benchmark Report: Custom Metal Engine vs. Ollama & 1-Bit Ternary

## Executive Summary
This report presents an empirical, apples-to-apples performance evaluation comparing:
1. **Ollama (`llama3.2:1b`, Q8_0, 1.32 GB)** on its default macOS runner.
2. **Our Custom Metal Server on the EXACT SAME `llama3.2:1b` (Q8_0) weights**.
3. **Our Bonsai 1.7B Ternary Engine (`Ternary-Bonsai-1.7B`, PQ2_0, 441 MB)**.

Each configuration was evaluated across **100 standardized prompts** in full compute isolation on Apple Silicon (M4).

---

## Performance Summary Table

| Metric | Ollama (`llama3.2:1b`) | Our Metal Engine (`llama3.2:1b`) | Bonsai 1.7B (`PQ2_0` Ternary) |
| :--- | :---: | :---: | :---: |
| **Model Weight File** | `llama3.2:1b` (Q8_0) | **`llama3.2:1b` (Q8_0)** | `Ternary-Bonsai-1.7B` (PQ2_0) |
| **Weight Equality** | Baseline | **IDENTICAL WEIGHT FILE** | 1-bit Ternary Quantization |
| **Model Size in RAM** | 1,321 MB | 1,321 MB | **441 MB (-66.6%)** |
| **Active Parameters** | 1.23 Billion | 1.23 Billion | **1.72 Billion (+39.8%)** |
| **Prompts Evaluated** | 100 / 100 | 100 / 100 | 100 / 100 |
| **Mean Decode Speed** | 50.89 tok/s | **67.08 tok/s (+31.8%)** | **134.02 tok/s (2.63x vs Ollama)** |
| **Median Decode Speed** | 50.81 tok/s | **67.41 tok/s** | **133.59 tok/s** |
| **P95 Decode Speed** | 57.63 tok/s | **68.07 tok/s** | **141.07 tok/s** |
| **Max Peak Speed** | 59.26 tok/s | **68.43 tok/s** | **143.20 tok/s** |
| **Mean Prefill Latency** | 39.5 ms | **32.7 ms** | **45.0 ms** |

---

## Architectural Findings
1. **Direct Metal GPU Dispatch (+31.8%):** On identical 8-bit model weights, bypassing Ollama's Go runtime and running natively on custom Metal shaders yields a +31.8% throughput increase (67.08 vs 50.89 tok/s).
2. **Ternary Memory Bus Bandwidth (2.63x):** Packing 1.72B weights into 441 MB cuts memory bus churn by 66.6%, boosting generation to 134.02 tok/s on Apple Silicon unified memory.
