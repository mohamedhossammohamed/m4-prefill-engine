# Core Architecture & The 4-Brick Pipeline

This document provides a deep architectural walkthrough of the `m4-prefill-engine` execution pipeline. It details how the engine bypasses high-level framework abstractions to convert memory-bound Transformer prefill into a hardware-saturated, compute-bound operation on Apple Silicon.

---

## 1. Architectural Overview: The 4-Brick Philosophy

Standard LLM inference on Apple Silicon treats GPU execution as a sequence of isolated framework calls (GEMM $\to$ activation $\to$ transpose $\to$ attention $\to$ layernorm). Under this conventional pattern, prefill becomes severely memory-bound due to repeated DRAM roundtrips and LSU queue underutilization.

The `m4-prefill-engine` replaces this fragmented model with four tightly coupled execution "Bricks":

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                THE 4-BRICK PIPELINE                              │
 └──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │   BRICK 1    │ ──► │   BRICK 2    │ ──► │   BRICK 3    │ ──► │   BRICK 4    │
 │ Hardware MMA │     │ 128-bit LSU  │     │  Dual-SIMD   │     │  Barrier-Free│
 │ Coprocessor  │     │ 2D Swizzling │     │ SwiGLU Fusion│     │FlashAttention│
 │(16.8 TFLOPS) │     │ Padded SRAM  │     │(Zero DRAM MLP│     │(Q8_0 KV Comp)│
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. Brick 1: Hardware Matrix Coprocessor (AMX / MMA)

### The Problem
Traditional Metal shaders and early open-source Apple Silicon kernels rely on Vector ALUs executing packed SIMD operations:
```metal
// Vector ALU Baseline (Peak ~7.4 TFLOPS on Apple M4)
half4 p = fma(a_vec, b_vec, c_vec);
```
While portable, Vector ALUs leave Apple Silicon's dedicated Hardware Matrix Coprocessor idle, artificially capping compute density.

### The Implementation
Brick 1 binds execution directly to Apple's matrix coprocessor fragments using Metal's `simdgroup_matrix<T, M, N>` intrinsics.

*   **Matrix Tile Size:** `simdgroup_matrix<half, 8, 8>` fragments.
*   **Accumulation Precision:** `simdgroup_matrix<float, 8, 8>` (FP32 accumulator fragments ensure zero underflow during large inner products).
*   **Threadgroup Allocation:** A single SIMDgroup (32 threads) cooperative execution tile of $32 \times 32$ or a 4-SIMDgroup tile of $64 \times 64$.

```metal
// 1. Fragment Declaration (4x4 grid of 8x8 fragments per SIMDgroup)
simdgroup_matrix<float, 8, 8> acc[4][4];
#pragma unroll
for (int r = 0; r < 4; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

// 2. Fragment Load from Padded Threadgroup Memory (SRAM)
simdgroup_matrix<half, 8, 8> a_frag[4];
simdgroup_matrix<half, 8, 8> b_frag[4];

#pragma unroll
for (int r = 0; r < 4; r++) {
    simdgroup_load(a_frag[r], &sh_A[(sg_row_offset + r * 8) * 36 + k_off], 36);
}

#pragma unroll
for (int c = 0; c < 4; c++) {
    simdgroup_load(b_frag[c], &sh_B[k_off * 64 + (sg_col_offset + c * 8)], 64);
}

// 3. Hardware SIMDgroup Multiply-Accumulate
#pragma unroll
for (int r = 0; r < 4; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        simdgroup_multiply_accumulate(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }
}
```

### Silicon Impact
*   **Hardware Ceiling:** Scales peak compute throughput from ~7.4 TFLOPS (Vector ALU) to **16.8 TFLOPS** (Matrix Coprocessor) on Apple M4.
*   **Instruction Efficiency:** Replaces 64 scalar FMA instructions with 4 hardware matrix instructions per inner loop step.

---

## 3. Brick 2: 128-bit LSU Saturation, 2D Swizzling & Padded SRAM

### The Problem: SRAM Bank Conflicts & Memory Stalls
Apple Silicon GPUs utilize a banked threadgroup SRAM architecture (typically 32 banks, 4 bytes wide). 
When multiple threads in a SIMDgroup read identical column strides from shared memory, or when memory accesses cross misaligned cachelines, the hardware serializes accesses into multiple cycles (bank conflicts).

### The Implementation: 3-Layer Ingestion Pipeline

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      BRICK 2 MEMORY INGESTION PIPELINE                      │
 └─────────────────────────────────────────────────────────────────────────────┘
   DRAM (128-bit Coalesced)     SRAM Padded Stride [64][36]     MMA Core (1-Cycle)
 ┌──────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────┐
 │ float4 / half8 Vector    │───►│ Stride 36 (32 + 4 pad)  │───►│ Broadcast to    │
 │ In-Flight Load Queue (LSU│    │ Guarantees Zero Bank    │    │ 8x8 simdgroup   │
 │ Full Saturation)         │    │ Conflicts across 32-T   │    │ matrix fragments│
 └──────────────────────────┘    └─────────────────────────┘    └─────────────────┘
```

1. **128-bit Vector Firehose:** All global DRAM reads from activation buffer $A$ are cast to `float4` (128-bit loads), perfectly aligning with M4's 16-byte memory bus granularity.
2. **Padded Shared Memory (`[64][36]`):** Shared memory tiles are allocated with a pitch of 36 elements rather than 32:
   $$\text{Memory Address} = \text{row} \times 36 + \text{col}$$
   The $+4$ padding ensures that each subsequent row shifts its starting memory bank by 4, completely eliminating column-read bank conflicts during `simdgroup_load`.
3. **Double-Buffered Ping-Pong Staging:** Uses two asynchronous buffers (`sh_A[2]`, `sh_B[2]`) to overlap global memory prefetching of iteration $k+1$ with MMA computation of iteration $k$.

```metal
// Double-buffered SRAM allocation with stride 36 padding
threadgroup half (*sh_A)[64][36] = (threadgroup half (*)[64][36])shmem;
threadgroup half (*sh_B)[32][64] = (threadgroup half (*)[32][64])(shmem + 4608);

// 128-bit Coalesced Load
float4 val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
*reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c])     = val.xy;
*reinterpret_cast<threadgroup float2*>(&sh_A[buf_idx][r][c + 4]) = val.zw;
```

---

## 4. Brick 3: Dual-SIMDgroup SwiGLU Fusion

### The Problem: The DRAM MLP Bottleneck
In modern LLMs (e.g., LLaMA 3.1 8B, $K=4096, N_{\text{mlp}}=14336$), the MLP block consists of three projections:
1. $\text{Gate} = X \cdot W_{\text{gate}}$
2. $\text{Up} = X \cdot W_{\text{up}}$
3. $\text{Act} = \text{SiLU}(\text{Gate}) \odot \text{Up}$
4. $\text{Down} = \text{Act} \cdot W_{\text{down}}$

Conventional runtimes execute Gate and Up projections as two distinct GEMM kernels, writing intermediate activations ($X \cdot W_{\text{gate}}$ and $X \cdot W_{\text{up}}$) back to DRAM, only to immediately reload them for the SwiGLU activation kernel. For an 8B model with $M=2048$, this wastes **7.68 GB of DRAM bandwidth per layer**.

### The Implementation: Dual-SIMD Collaborative Execution
Brick 3 fuses Gate projection, Up projection, and the SwiGLU activation into a single kernel:
*   **SIMDgroup 0 & 1:** Compute the Gate projection tile ($32 \times 64$).
*   **SIMDgroup 2 & 3:** Compute the Up projection tile ($32 \times 64$).
*   **Shared Activation Staging:** Input activation $A$ ($M \times K$) is loaded into shared SRAM **once** and broadcast to all 4 SIMDgroups.
*   **In-Register SwiGLU Reduction:** As soon as Gate and Up accumulators finish, SIMDgroups exchange fragments in SRAM, evaluate the exact SiLU activation:
    $$\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$$
    and compute the Hadamard product in registers before writing directly to the output buffer.

```metal
// In-Register SwiGLU Activation (Zero DRAM Roundtrips)
float gate_val = (float)sh_Gate[r][c];
float up_val   = (float)sh_Up[r][c];
float silu_val = gate_val / (1.0f + exp(-gate_val));
float out_val  = silu_val * up_val;
Out[global_r * N_mlp + global_c] = (half)out_val;
```

---

## 5. Brick 4: Barrier-Free FlashAttention & Dynamic Q8_0 KV Cache

### The Problem: Attention Synchronization Overhead
Standard FlashAttention implementations rely heavily on `threadgroup_barrier(mem_flags::mem_threadgroup)` inside the inner softmax loop to compute row-wise maximums ($m_i$) and normalizers ($l_i$). On Apple Silicon, frequent threadgroup barriers stall SIMD execution pipelines and drain instruction caches.

### The Implementation: Register Butterfly Trees + Dynamic Q8_0 KV
Brick 4 introduces two architectural enhancements:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │               BRICK 4 BARRIER-FREE FLASHATTENTION ARCHITECTURE              │
 └─────────────────────────────────────────────────────────────────────────────┘
      SIMD Butterfly Reduction                    Dynamic Q8_0 KV Cache
 ┌────────────────────────────────┐         ┌─────────────────────────────────┐
 │ Lane 0-31 register exchange    │         │ 8-bit symmetric quantization    │
 │ via simd_shuffle_down()        │         │ 34 bytes per 32 elements        │
 │ Zero threadgroup barriers      │         │ 47% reduction in memory volume  │
 │ in online softmax inner loop   │         │ Aligned float4 vector unpacking │
 └────────────────────────────────┘         └─────────────────────────────────┘
```

#### 1. Barrier-Free Online Softmax Reductions
Online running softmax state ($m_{\text{new}}, l_{\text{new}}$) is updated using intra-warp register shuffle instructions (`simd_shuffle_down`):

```metal
// SIMDgroup Register Butterfly Reduction (Zero Shared Memory, Zero Barriers)
inline float simd_max_reduce(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return simd_broadcast_first(val);
}
```

#### 2. Dynamic Q8_0 KV Cache Compression
Key and Value tensors are dynamically quantized to 8-bit symmetric blocks (`block_q8_0`):
$$\text{Memory per 32 elements} = 32 \times 1\text{ byte (int8)} + 2\text{ bytes (FP16 scale)} = 34\text{ bytes}$$
*   **Footprint Reduction:** Reduces KV cache memory volume from 64 bytes (FP16) to 34 bytes per 32 elements (**46.88% bandwidth reduction**).
*   **Causal Skipping:** Skips full upper-triangular tiles when $\text{tile\_col} \times B_C > \text{max\_row}$.

---

## 6. Full Layer Integration & Kernel Dispatch

When all 4 Bricks are composed into a unified transformer layer forward pass (`unified_prefill_engine`):
1. **Input Normalization:** RMSNorm applied in registers.
2. **Fused QKV Projection (Brick 1 & 2):** Input activations multiplied by Q, K, V weights simultaneously using Direct-Head output layouts.
3. **FlashAttention (Brick 4):** Fused 2D BlockMMA attention with dynamic Q8_0 KV caching.
4. **Out-Projection (Brick 1):** Dense linear projection with residual accumulator addition.
5. **Fused SwiGLU MLP (Brick 3):** Gate/Up dual-SIMD projection + in-register SiLU + Down projection.

This complete composition eliminates memory bottlenecks and anchors LLM prefill firmly toward the Apple M4 Matrix Coprocessor's theoretical hardware peak of ~16.8 TFLOPS.

---

## 7. Modular Decoupled MSL Architecture (v0.2.3)

To allow experimental extensions and clean separation of concerns without duplicating thousands of lines of AMX matrix loops, the engine introduces a decoupled header-only MSL library (`include/metal/`):

```
  ┌─────────────────────────────────────────────────────────────┐
  │         Layer Coordinator (TransformerLayerCoordinator)     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
     ┌───────────────────────────┼────────────────────────────┐
     ▼                           ▼                            ▼
┌──────────────────┐   ┌───────────────────┐        ┌───────────────────┐
│ gemm_mma.metal   │   │ swiglu_dual_simd  │        │ flash_attention   │
│ <TCodec, DIRECT> │   │ <TCodec>          │        │ <HeadDim D>       │
└────────┬─────────┘   └─────────┬─────────┘        └─────────┬─────────┘
         │                       │                            │
         └───────────────────────┼────────────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ Quantization Codecs (TCodec)  │
                 │ • q4_0.metal                  │
                 │ • mlx_4bit.metal              │
                 │ • q4_k.metal                  │
                 │ • ternary_1_58.metal          │
                 │ • var_rate_affine.metal       │
                 │ • exl3.metal                  │
                 └───────────────────────────────┘
```

### Key Modular Components

1. **Templated BlockMMA Core (`include/metal/ops/gemm_mma.metal`):**
   ```metal
   template <typename TCodec, bool DIRECT_HEAD_ROUTING = false>
   inline void block_mma_64x64_gemm_core(...) { ... }
   ```
   Parameterized by `TCodec`, which only requires a single static function:
   `TCodec::unpack_column(B, col, kb, K, sh_B, linear_tid)`.
   The outer-product AMX coprocessor matrix accumulation, double-buffered threadgroup SRAM staging, and writeback logic are implemented once and reused across all formats.

2. **Dual-SIMD Cooperative SwiGLU Core (`include/metal/ops/swiglu_dual_simd.metal`):**
   ```metal
   template <typename TCodec>
   inline void swiglu_mma_dual_simd_core(...) { ... }
   ```
   Evaluates Gate and Up projections concurrently in threadgroup registers using `TCodec`, computes in-SRAM $\text{SiLU}(\text{Gate}) \times \text{Up}$ activation, and writes directly to DRAM with zero intermediate global memory footprint.

3. **Barrier-Free FlashAttention Core (`include/metal/ops/flash_attention.metal`):**
   ```metal
   template <uint D>
   inline void flash_attn_mma_64x64_fp16_core(...) { ... }
   ```
   Parameterized on head dimension $D \in \{64, 128\}$. Implements online softmax tracking and tensor-core accumulation using intra-warp register butterfly reductions (`simd_shuffle_down`).

4. **Runtime Shader Preprocessor (`core/metal/shader_loader.mm`):**
   Transparently resolves relative `#include "..."` directives and `#pragma once` guards at runtime, creating self-contained compilation translation units for `newLibraryWithSource:options:error:` without external build-time preprocessors.
