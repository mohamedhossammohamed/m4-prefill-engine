# Changelog

All notable changes to the `m4-prefill-engine` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2026-09-06 - Metal vs. Ollama & Ternary Serving

### Summary
Version 0.3.1 packages an empirical, 100-prompt comparative metrology harness evaluating native Apple Silicon Metal GPU serving against stock Ollama runners across identical model weights (`llama3.2:1b` Q8_0) and hardware-accelerated 1-bit ternary formats (`Ternary-Bonsai-1.7B` PQ2_0). On identical model weights, the custom Metal server outperforms Ollama by +31.8% in decode throughput (67.08 vs 50.89 tok/s) and delivers 17% lower prefill latency, bypassing runtime abstraction layers. Transitioning to 1-bit ternary cuts memory bus bandwidth by 66.6% (441 MB vs 1.32 GB), achieving 134.02 tok/s (2.63x faster than Ollama). Standardized scripts provide one-command server runners, automated language coherence verification (Python AST compilation and scientific synthesis), and a self-contained project architecture in `projects/bonsai/`.

### Added
* **100-Prompt Comparative Metrology Suite (`benchmarks/`):**
  * Automated benchmark CLI `benchmarks/bench_metal_vs_ollama.py` evaluating identical prompts across Ollama, custom Metal GPU server, and Bonsai 1.7B ternary engine in strict single-process compute isolation.
  * Raw empirical telemetry preserved in `benchmarks/logs/bench_metal_vs_ollama.json` and formatted report in `benchmarks/logs/bench_metal_vs_ollama_report.md`.
  * Measured statistical performance: Ollama (50.89 tok/s mean, 50.81 tok/s median), Metal on identical weights (67.08 tok/s mean, 67.41 tok/s median, +31.8%), Bonsai 1.7B ternary (134.02 tok/s mean, 133.59 tok/s median, 2.63x vs Ollama).
* **Automated Language Coherence & AST Test Harness (`scripts/`):**
  * `scripts/test_llama32_language.py`: Verifies non-gibberish generation, validates generated Python code via `ast.parse()`, and checks multi-step scientific reasoning against degeneration.
  * `scripts/test_metal_server.sh`: Automated launcher for `llama-server` with `-ngl 99` and FlashAttention on isolated test port 8089.
  * `scripts/run_ollama_server.sh`: Daemon runner verifying local Ollama installation, model pull, and API readiness.
* **Modular Self-Contained Project (`projects/bonsai/`):**
  * Dedicated directory isolating Bonsai ternary serving (`run.sh`, `server.py`, `benchmark.py`, `api_key.txt`, and ARM NEON runtime) with zero-copy symlinks to local GGUF weights.
* **Build System Additions (`Makefile`):**
  * Phony targets `make test_metal`, `make bench_ollama`, and `make test_language`.

---

## [0.3.0] - 2026-09-04 - Warming the Tires

### Summary
Version 0.3.0 transitions `m4-prefill-engine` from a standalone prefill micro-architecture into a complete, modular, end-to-end inference engine capable of prompt prefill ($M \ge 1$) and autoregressive token generation ($M = 1$) on Apple Silicon. A zero-copy Metal Unified Memory Architecture (UMA) bridge allows MLX tensors to be consumed directly by custom Metal shaders without memory copies. The transformer stack is fully decoupled into swappable building blocks (`src/engine/`) supporting arbitrary model topologies (including Gemma 2 configurations). The quantization catalog expands to 8 formats with the addition of the PrismML Q2_0 128-weight ternary codec and a native endian-safe GGUF loader. Automated comparative benchmarks evaluate custom hardware execution head-to-head against stock MLX baselines.

### Added
* **Zero-Copy Metal UMA C-ABI & Python Bridge (`core/bridge/`):**
  * Dynamic library `libm4_bridge.dylib` providing zero-copy buffer sharing between MLX arrays and Metal device memory via `newBufferWithBytesNoCopy`.
  * Python interface `MetalUMABridge` with buffer lifetime tracking, automated contiguous memory alignment, and non-finite (NaN/Inf) tripwires.
  * Hardware-accurate physical memory tracking using Mach kernel `task_vm_info.phys_footprint`.
* **Decoupled Unified Inference Engine (`src/engine/`):**
  * `M4QuantizedLinear`: Quantized linear layer supporting dynamic dispatch between GEMV ($M=1$) and GEMM ($M > 1$) across all supported codecs with support for arbitrary leading batch dimensions.
  * `M4KVCache`: Dual-mode KV cache supporting pre-allocated circular DRAM buffers (`in_ram`) or Direct I/O NVMe flash streaming (`out_of_core`) with `F_NOCACHE` and zero disk litter.
  * `TransformerBlock` & `TransformerModel`: Composable architecture supporting Grouped-Query Attention (GQA), causal masking, RoPE sequence offset alignment, SwiGLU / GeGLU activations, and Gemma 2 `gemma_add_one` RMSNorm.
  * `InferenceEngine`: High-level runtime managing prompt prefill, token decode loops, greedy argmax / temperature / top-p sampling, and early EOS termination.
* **PrismML Q2_0 Ternary Codec (`QUANT_PRISM_Q2_0`):**
  * 128-weight ternary quantization block (`block_prism_q2_0`, 34 bytes per block, 2.125 bpw) using 2-byte FP16 scale and 32 bytes packed 2-bit codes.
  * Modular MSL unpacker in `include/metal/quant/prism_q2_0.metal` consuming the central `block_mma_64x64_gemm_core` with zero modifications to shared dispatch code.
  * Spec-compliant dequantization $w = (q - 1) \times d$, mapping reserved code $q=3$ strictly to $+2 \times d$.
* **Native Endian-Safe GGUF File Loader (`core/weights/`):**
  * Zero-copy `mmap` binary parser validating GGUF v2/v3 magic and headers, metadata KV table, tensor-info table, and alignment padding.
  * Endian-safe extraction of Q2_0 tensor weights directly into `block_prism_q2_0` structures.
* **Automated Comparative Benchmark Harness (`benchmarks/`):**
  * CLI tool `bench_m4_vs_mlx.py` evaluating custom M4 hardware pipelines against stock MLX baselines across Time to First Token (TTFT), prefill throughput, decode latency, decode throughput, and active UMA footprint.
  * 32MB direct I/O cache purge (`purge_cold_caches`) preventing SLC and buffer cache contamination.
  * Pipelined asynchronous command-buffer encoding eliminating per-projection CPU wait bubbles.
* **Quantization Registry Extension:**
  * Extended `QuantRegistry` to 8 built-in formats (`Q4_0`, `MLX_4BIT`, `Q4_K`, `TERNARY_1_58`, `VAR_RATE_AFFINE`, `EXL3`, `Q8_0`, `PRISM_Q2_0`).

### Verification & Test Coverage
* **Native C++ Test Suite (`make test`):**
  * 7 test targets passing cleanly under `clang++ -O3 -Wall` with zero warnings.
  * Bit-exact parity verified across 35 configurations (7 codecs $\times$ 5 token boundaries, `diff == 0.000000`).
  * 54/54 E2E tier-1 feature tests passed.
  * 8/8 GGUF loader integrity tests passed.
* **Python Engine & Red-Team Suites:**
  * 42 verification probes across bridge, module, orchestrator, and benchmark harnesses passing with zero detected defects.

---

## [0.2.3] - 2026-09-03 - More Flexibility, Smaller Legos

### Summary
Version 0.2.3 modularizes the Metal quantization and inference pipeline into composable, isolated components without altering runtime numerical behavior or hardware execution paths. Format-specific dequantization logic for 6 codecs is isolated into per-codec MSL headers under `include/metal/quant/`, feeding unified templated `TCodec` operator cores for BlockMMA GEMM, Dual-SIMD SwiGLU, and FlashAttention in `include/metal/ops/`. A host-side dynamic `QuantRegistry` enables Open-Closed format addition without touching core matrix multiplication kernels. All refactored kernels are validated bit-exact against monolithic baselines across 30 configurations (6 codecs $\times$ 5 boundary token counts), backed by a 204-test automated verification suite (`make test`).

### Added
* **Modular Codec Headers (`include/metal/quant/`):**
  * Extracted 6 quantization codec unpackers (`q4_0.metal`, `mlx_4bit.metal`, `q4_k.metal`, `ternary_1_58.metal`, `var_rate_affine.metal`, `exl3.metal`) behind a uniform `TCodec::unpack_column` interface contract (`codec_traits.metal`).
* **Templated Operator Cores (`include/metal/ops/`):**
  * `block_mma_64x64_gemm_core<TCodec, DIRECT>` (`gemm_mma.metal`): Unified 2D AMX tensor core matrix multiplication engine shared across all quantization formats.
  * `swiglu_mma_dual_simd_core<TCodec>` (`swiglu_dual_simd.metal`): Cooperative dual-SIMD Gate/Up projection and SiLU activation engine.
  * `flash_attn_mma_64x64_fp16_core<D>` (`flash_attention.metal`): Templated barrier-free online softmax FlashAttention core ($D \in \{64, 128\}$).
  * Runtime recursive MSL `#include` and `#pragma once` preprocessor (`core/metal/shader_loader.mm`).
* **Dynamic Codec Registry (`src/router/quant_registry.h`):**
  * Declarative `QuantRegistry` and `REGISTER_QUANT_CODEC` macro enabling runtime discovery and dispatch across all 7 formats plus custom runtime formats (`QUANT_CUSTOM`).
* **Bit-Exact Pre/Post Parity Suite (`tests/test_kernel_parity.mm`):**
  * Automated side-by-side GPU execution verifying bit-for-bit output identity (`diff == 0.000000`) between modular headers and monolithic kernels across 30 configurations ($M \in \{33, 127, 128, 129, 512\}$).
* **Composable Transformer Layer Coordinator (`models/transformer_layer.h`):**
  * Modular coordinator binding RMSNorm, QKV GEMM, FlashAttention, Residual Adds, and SwiGLU forward passes into unified command buffers with GPU verification (`tests/test_transformer_layer.mm`).
* **Metrological Invariants Suite (`core/`):**
  * Authoritative implementations of 16KB Direct I/O memory alignment (`core/memory/page_allocator.mm`), Mach kernel `phys_footprint` tracking with zero-leak verification (`core/memory/uma_tracker.mm`), 32MB SLC cache flushing (`core/memory/cache_flush.mm`), non-finite tripwires (`core/metrology/tripwires.mm`), and cognitive telemetry formatting (`core/metrology/telemetry_format.mm`).
* **Automated Verification Harness (`make test`):**
  * 204 test cases across 6 test binaries verifying core invariants (108 tests), Metal header compilation and static register capacity (10 tests), QuantRegistry full 7-format coverage (5 tests), pre/post refactor bit-exact kernel parity across all formats and boundary token counts (30 tests), composable transformer layer coordinator (1 test), and E2E feature coverage for $M \in \{33, 127, 128, 129, 2048\}$ (50 tests). Zero memory leaks.

---

## [0.2.1] - 2026-09-03 - Beyond MLX Prefill Speeds

### Summary
Version 0.2.1 ("Beyond MLX Prefill Speeds") packages the architectural optimizations, memory layout breakthroughs, and threadgroup SRAM bank conflict eliminations that enable `m4-prefill-engine` to surpass Apple MLX's native Metal engine in prefill latency on Apple Silicon (achieving **1.18x faster** prefill at $M=2048$ tokens and **1.16x faster** at $M=33$ tokens on the 1B tier, while beating `llama.cpp` by **1.67x–2.30x** across all sequence lengths).

### Key Architectural Speed Innovations (The "Beyond MLX" Suite)
* **128-Bit LSU-Aligned Planar Quantized Layout (`mlx_4bit_planar`):**
  * Eliminated the interleaved 20-byte struct stride ($20 \pmod{16} = 4$) by deinterleaving MLX 4-bit weights into contiguous 16-byte aligned planar arrays.
  * Replaced unaligned scalar byte reads with direct 128-bit vector loads (`uint4`), saturating Apple Silicon's 128-bit Load-Store Unit (LSU) with zero split-load cache penalties.
* **Threadgroup SRAM Stride-36 Bank Padding:**
  * Padded `sh_A` matrix tiles from stride 32 to stride 36 (`half sh_A[64][36]`), completely eradicating 32-way shared memory bank conflicts across Apple Silicon's 32-bank threadgroup SRAM during Hardware Matrix Coprocessor (`simdgroup_matrix`) outer-product execution.
  * Solved the power-of-2 bank aliasing bottleneck that degraded saturated sequence lengths ($M=2048$), propelling full-model 1B prefill from 2.02 s (MLX) down to **1.71 s**.
* **Direct-Head Multi-Head Attention Layout:**
  * Eliminated post-GEMM memory transpositions by directly projecting Q, K, and V into $[H, M, D]$ layout, bypassing round-trip DRAM churn prior to FlashAttention.
* **Dual-SIMDgroup SwiGLU Cooperative Fusion:**
  * Evaluated Gate and Up projections concurrently in threadgroup registers with zero round-trip global DRAM writes, saving 7.68 GB of DRAM traffic per layer on 8B models.

### Metrology & Audit Remediation
* **Zero Fabricated Verification Telemetry:** Purged all hardcoded `"0.000000"` string literals from `bench_streaming_kv.mm` and uncomputed `0.000000` diffs from `bench_all_scales.mm`.
* **Transparent Verification Delineation:** Real computed CPU validation is reported for verifiable token ranges ($M \le 128$, $\text{MaxDiff} \le 0.000977$), while larger sweeps ($M > 128$) are transparently labeled `N/A (CPU Gold Reference Gated)` with verified GPU invariant assertions (zero NaN/Inf).
* **Release Scope Containment:** Formally designated out-of-core flash streaming (1M context) and speculative decoding as experimental research prototypes excluded from the v0.2.1 release deliverables.

---

## [0.2.0] - 2026-09-02

### Summary
Version 0.2 transforms `m4-prefill-engine` from a specialized prefill micro-benchmark into a complete, hardened, unified inference architecture for Apple Silicon. This release introduces universal multi-format quantization routing, out-of-core 1,000,000-token flash streaming, speculative burst decode off NVMe storage, and a complete overhaul of measurement and verification metrology.

### Added
* **Hardware Matrix Coprocessor (`simdgroup_matrix`) Integration:** Transitioned core GEMM matrix operations from Vector ALUs (`fma(half4)`, ~7.4 TFLOPS peak) to Apple's Hardware Matrix Units (`simdgroup_matrix<half, 8, 8>`, ~16.8 TFLOPS peak), achieving up to **12.65 TFLOPS (75.8% of theoretical peak)**.
* **Universal Quantization Router:** Modular on-the-fly dequantization router supporting 6 distinct formats in a single unified pipeline:
  * `QUANT_Q4_0` (GGUF standard 32-element symmetric)
  * `QUANT_MLX_4BIT` (MLX packed uint32 with FP16 scale + bias)
  * `QUANT_Q4_K` (GGUF 256-element super-blocks with 6-bit scales/mins)
  * `QUANT_VAR_AFFINE` (Variable-rate asymmetric affine quantization)
  * `QUANT_EXL3` (Hierarchical 3-bit codebook vector quantization)
  * `QUANT_TERNARY_1_58` (BitNet-style 2-bit packed ternary weights in $\{-1, 0, +1\}$)
* **1,000,000-Token Out-of-Core SSD Flash Streaming Engine:**
  * Direct PCIe NVMe I/O using `fcntl(fd, F_NOCACHE, 1)` with 16KB system page-aligned buffers (`posix_memalign`) to bypass the macOS Unified Buffer Cache (UBC), sustaining 2.0–3.0 GB/s physical storage throughput.
  * Dual 128MB asynchronous ring buffer queue (`AsyncKVRingBuffer`) overlapping SSD DMA with GPU Matrix Coprocessor computation.
  * Multi-chunk online softmax state persistence and rescaling induction ($\text{MaxDiff} \le 0.000977$).
  * Zero-Disk-Litter protocol (`unlink()` immediately upon descriptor `open()`) ensuring zero storage leaks on exit.
* **On-the-Fly Speculative Burst Decode ($K=64$):**
  * Parallel speculative burst verification evaluating 64 candidate draft tokens in parallel matrix registers over a single streaming pass of the 1M context.
  * Transforms out-of-core flash generation from memory-bound (0.60 tok/s) into compute-bound (**35.2 measured tok/s**, 36.8–37.2 GPU compute tok/s at 1M tokens).
* **Dual-SIMDgroup Cooperative SwiGLU Engine (Brick 3):**
  * Cooperative SIMDgroup work partitioning (SIMD 0/1 on Gate, 2/3 on Up) sharing input activation $X$ in on-chip SRAM.
  * In-SRAM $\text{SiLU}(\text{Gate}) \times \text{Up}$ epilogue fusion, saving **7.68 GB DRAM traffic** per layer on 8B models while keeping register pressure at ~26 regs/thread (100% GPU occupancy).
* **2D BlockMMA Tensor-Core FlashAttention (Brick 4):**
  * Online softmax reduction via intra-warp butterfly shuffle trees (`simd_shuffle_down`) replacing threadgroup memory barriers.
  * Causal triangular block skipping saving ~50% arithmetic.
  * Dynamic Q8_0 KV Cache compression with on-the-fly SIMD dequantization (50% reduction in KV memory bandwidth).
* **Sustained Passive Thermal Stress Test:** 60-second continuous heat-soak benchmark measuring thermal throttling (11.8% drop, 3.2% latency coefficient of variation) on fanless M4 chassis.
* **Comprehensive Documentation Suite (`docs/`):** 7 in-depth technical guides covering Quantized GEMM, SwiGLU, FlashAttention, 1M SSD Streaming, Universal Router, Cross-Metal Porting Guide (CUDA/ROCm/Vulkan/WebGPU), and Master Benchmark Telemetry.

### Changed & Hardened (Metrology & Metrology Audits)
* **16KB Page Alignment for Direct I/O:** Replaced unaligned `std::vector` staging buffers with 16KB page-aligned `posix_memalign` buffers, eliminating silent macOS kernel fallbacks to buffered I/O.
* **Unified Buffer Cache (UBC) Purge:** Added mandatory 32MB direct I/O dummy file eviction before cold-cache benchmark runs to prevent OS cache contamination.
* **UMA Memory Measurement:** Replaced `mach_task_basic_info.resident_size` with `task_vm_info.phys_footprint` to capture physical allocations across Metal shared memory buffers (`MTLResourceStorageModeShared`).
* **Dual Latency Telemetry:** Updated all benchmark output tables to report both **End-to-End Latency** (including host NVMe prefetch I/O) and **GPU Compute Only Latency**.
* **Honest Verification Labeling:** CPU double-precision verification is locked for $M \le 2048$. Output for $M > 2048$ is explicitly labeled `[NOT VERIFIED — CPU gold infeasible at this scale]` with explicit notes on the quadratic $O(M^2)$ CPU runtime.
* **Strict NaN/Inf Error Assertions:** Updated all validation loops to hard abort immediately upon encountering any non-finite float value.

---

## [0.1.0] - 2026-08-15

### Initial Prototype Release
* Initial implementation of custom Metal kernels for Q4_0 quantized matrix multiplication on Apple M4.
* 1D Vector ALU scalar loops (`half4 fma`) for GEMM projections.
* Basic FlashAttention prototype with FP16 KV cache.
* Basic benchmark suite comparing against llama.cpp on prompt lengths $M \in [128, 512, 1024, 2048]$.
* Proved feasibility of saturating Apple Silicon memory bandwidth with custom Metal shaders.
