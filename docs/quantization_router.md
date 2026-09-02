# Universal Quantization Router: Formats, Bitstreams & Layouts

This document provides a comprehensive technical reference for the **Universal Quantization Router** in `m4-prefill-engine`. It details the binary layout, unpacker implementations, SRAM staging strategies, layout routing transforms, and empirical hardware findings across six supported quantization formats.

---

## 1. Architectural Motivation

Open-source LLM inference on Apple Silicon has historically suffered from format fragmentation:
*   **MLX** exclusively optimizes its proprietary affine 4-bit layout (`MLX_4BIT`).
*   **llama.cpp** focuses on GGUF quantization formats (`Q4_0`, `Q4_K`, `Q8_0`).
*   **ExLlamaV2/V3** utilizes variable bit-depth affine and hierarchical codebooks on CUDA GPUs.
*   **BitNet** introduces 1.58-bit ternary representations ($\{-1, 0, +1\}$).

The **Universal Quantization Router** decouples the physical weight format from the computational core. It decodes all six formats on-the-fly inside high-speed threadgroup SRAM (`sh_B[32][64]`), feeding a single unified 2D BlockMMA execution engine.

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                       UNIVERSAL QUANTIZATION ROUTING MATRIX                      │
 └──────────────────────────────────────────────────────────────────────────────────┘
   Input Activation A [M, K]         Quantized Weight Storage B [N, K/Block]
 ┌───────────────────────────┐     ┌──────────────────────────────────────────────┐
 │ • Standard [M, K]         │     │ • Q4_0 (GGUF 32-elem block, 4.50 bpw)        │
 │ • Direct-Head [H, M, D]   │     │ • MLX 4-bit (64-elem affine, 5.00 bpw)       │
 │                           │     │ • Q4_K (GGUF 256-elem super-block, 4.50 bpw) │
 └─────────────┬─────────────┘     │ • Var-Rate Affine (3/4/5-bit, 5.00 bpw)      │
               │                   │ • EXL3 (Hierarchical codebook, 4.50 bpw)     │
               │                   │ • Ternary 1.58-bit (BitNet, 3.00 bpw)        │
               │                   └──────────────────────┬───────────────────────┘
               │                                          │
               ▼                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                    ON-THE-FLY UNPACKING & SRAM STAGING                           │
 │     Vectorized Bit-Shift Dequantization -> Threadgroup Memory sh_B[32][64]       │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                 UNIFIED 2D BLOCKMMA HARDWARE EXECUTION CORE                      │
 │    simdgroup_matrix<half, 8, 8> fragments -> 16.8 TFLOPS Hardware Coprocessor    │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Format Specifications & Memory Layouts

### 1. Q4_0 (GGUF Standard Symmetric 4-Bit)
*   **Block Size:** 32 elements.
*   **Weight Footprint:** 18 bytes per block (16 bytes packed nibbles + 2 bytes FP16 scale).
*   **Effective Bit-Depth:** 4.50 bits/weight.
*   **Dequantization Formula:**
    $$w_i = (q_i - 8) \times d$$

```metal
struct block_q4_0 {
    half d;          // FP16 Scale Factor (2 bytes)
    uint8_t qs[16];  // 32 packed 4-bit nibbles (16 bytes)
};

// Fast Bitwise Vector Unpacker:
inline void unpack_q4_0_block(block_q4_0 blk, thread half4 vl[4], thread half4 vh[4]) {
    half d = blk.d;
    half4 hd = half4(d);
    half4 h_off = half4(-8.0h * d);

    uint w0 = read_u32_unaligned(blk.qs + 0);
    uint w1 = read_u32_unaligned(blk.qs + 4);
    uint w2 = read_u32_unaligned(blk.qs + 8);
    uint w3 = read_u32_unaligned(blk.qs + 12);

    vl[0] = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
    vh[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[1] = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
    vh[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[2] = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
    vh[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
    vl[3] = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
    vh[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
}
```

---

### 2. MLX 4-Bit (Apple MLX Affine 4-Bit)
*   **Block Size:** 64 elements.
*   **Weight Footprint:** 40 bytes per block (32 bytes packed nibbles + 2 bytes FP16 scale + 2 bytes FP16 bias + 4 bytes padding).
*   **Effective Bit-Depth:** 5.00 bits/weight.
*   **Dequantization Formula:**
    $$w_i = q_i \times d + b$$

```metal
struct block_mlx_4bit {
    half d;          // FP16 Scale Factor (2 bytes)
    half b;          // FP16 Bias Factor  (2 bytes)
    uint8_t qs[32];  // 64 packed 4-bit nibbles (32 bytes)
    uint8_t pad[4];  // 16-byte struct alignment padding (4 bytes)
};
```

---

### 3. Q4_K (GGUF Super-Block Quantization)
*   **Block Size:** 256 elements (composed of eight 32-element sub-blocks).
*   **Weight Footprint:** 144 bytes per super-block (128 bytes nibbles + 4 bytes FP16 scales + 12 bytes 6-bit sub-block scales).
*   **Effective Bit-Depth:** 4.50 bits/weight.
*   **Dequantization Formula:**
    $$w_i = (q_i - 8) \times d_{\text{global}} \times s_{\text{sub}}$$

```metal
struct block_q4_k {
    half d;            // Global super-block scale
    half dmin;         // Global super-block minimum scale
    uint8_t scales[12];// Packed 6-bit sub-block scales
    uint8_t qs[128];   // 256 packed 4-bit nibbles
};
```

---

### 4. Grouped Variable-Rate Affine (EXL2-Style)
*   **Block Size:** 256 elements.
*   **Weight Footprint:** 160 bytes per super-block (mixed bit-depth streams: 3-bit, 4-bit, and 5-bit sub-blocks).
*   **Effective Bit-Depth:** 5.00 bits/weight.
*   **Architecture:** Grouped affine quantization with variable bit-depths. *(Note: This is grouped variable-rate affine quantization, not the ExLlamaV2 G-matrix codebook format).*

```metal
struct block_var_rate_affine {
    half scales[4];     // 4 FP16 sub-block scales (8 bytes)
    half biases[4];     // 4 FP16 sub-block biases (8 bytes)
    uint8_t bitstream[144]; // Packed variable-rate stream (3-bit/4-bit/5-bit)
};
```

---

### 5. EXL3 (Hierarchical Vector Codebook)
*   **Block Size:** 256 elements.
*   **Weight Footprint:** 144 bytes per super-block.
*   **Effective Bit-Depth:** 4.50 bits/weight.
*   **Architecture:** Hierarchical vector codebook centroids ($16 \times 4\text{D}$ codebook), 4-bit vector indices, sub-block residual corrections, and FP16 global scale/bias.

```metal
struct block_exl3 {
    half scale;         // Global FP16 scale
    half bias;          // Global FP16 bias
    uint8_t sub_scales[8]; // 8 sub-block scale multipliers
    uint8_t centroids[16]; // 16 vector centroid indices
    uint8_t indices[112];  // Packed 4-bit hierarchical vector indices
};
```

---

### 6. Ternary 1.58-Bit (BitNet / $1.58\text{ bpw}$)
*   **Block Size:** 32 elements.
*   **Weight Footprint:** 12 bytes per block (8 bytes packed ternary trits + 2 bytes FP16 scale + 2 bytes padding).
*   **Alphabet:** $\{-1, 0, +1\}$.
*   **Effective Bit-Depth:** 3.00 bits/weight (1.58 bits theoretical entropy).

```metal
struct block_ternary_1_58 {
    half scale;        // FP16 scale factor (2 bytes)
    uint8_t pad[2];    // 4-byte struct alignment padding (2 bytes)
    uint8_t qs[8];     // 32 packed 2-bit ternary trits (8 bytes)
};
```

---

## 3. The Ternary Reality on Apple Silicon (MMA vs. Vector ALU)

### The Theoretical Hypothesis
In BitNet literature, ternary quantization is celebrated for replacing expensive floating-point multiplications with pure additions and subtractions:
$$y = \sum_{w_i \in \{+1, -1, 0\}} x_i \cdot \text{sign}(w_i)$$

### The Empirical Finding
To test this on physical Apple Silicon, the router implements **two parallel execution paths**:
1. **`TERNARY_1_58_VEC (True Add/Sub Bypass)`:** Dequantizes trits and executes conditional branch-free vector additions/subtractions directly on Vector ALUs.
2. **`TERNARY_1_58_MMA (Bandwidth-Only FMA Fallback)`:** Unpacks ternary trits into shared memory `sh_B` and feeds them directly into the 16.8 TFLOPS Hardware Matrix Coprocessor (`simdgroup_multiply_accumulate`).

### Empirical Comparison on Apple M4 (LLaMA 8B Tier, $K=4096$)

| Sequence Length ($M$) | Vector ALU Add/Sub (`VEC`) | Hardware Matrix MMA (`MMA`) | Speedup of MMA over ALU |
| :---: | :---: | :---: | :---: |
| **$M = 33$** | 2.635 ms | **0.849 ms** | **3.10x faster** |
| **$M = 127$** | 5.273 ms | **2.065 ms** | **2.55x faster** |
| **$M = 128$** | 5.190 ms | **1.481 ms** | **3.50x faster** |
| **$M = 512$** | 19.914 ms | **5.669 ms** | **3.51x faster** |
| **$M = 2048$** | 79.178 ms | **21.960 ms** | **3.61x faster** |

### Why MMA Wins on Apple Silicon
*   **Compute Asymmetry:** Apple Silicon's Vector ALUs offer ~7.4 TFLOPS peak, whereas the Matrix Coprocessor delivers **16.8 TFLOPS**. Even with zero multiplications, Vector ALUs cannot overcome a $2.27\times$ raw compute deficit.
*   **Instruction Issue Bandwidth:** Packing and masking trits for ALU additions requires multiple integer bit-manipulation instructions per element, saturating the GPU execution pipeline.
*   **The True Value of Ternary on Apple Silicon:** The benefit of Ternary on M4 is **memory bandwidth**, not arithmetic bypass. At $3.0\text{ bpw}$, a complete 8B layer's weights require only **6.00 MB** (fitting entirely within the M4's 24MB System-Level Cache), completely eliminating DRAM traffic.

---

## 4. Direct-Head Layout Routing (`[M, K] -> [H, M, D]`)

Standard transformer engines project activations in flat row-major layout:
$$X_{\text{out}} \in \mathbb{R}^{M \times (H \cdot D)}$$
This requires a subsequent transpose kernel to partition activations into multi-head format:
$$X_{\text{head}} \in \mathbb{R}^{H \times M \times D}$$

### The Direct-Head Advantage
The Universal Quantization Router implements **Direct-Head Routing**:
*   The GEMM kernel writes directly into head-strided memory:
    $$\text{Global Address} = h \times (M \times D) + m \times D + d$$
*   **Zero Dynamic Transpositions:** Eliminates intermediate DRAM transpositions.
*   **Zero Dynamic Padding:** Handles arbitrary sequence lengths ($M=33, 127, 129, 2047$) with zero boundary padding overhead.

---

## 5. Comprehensive Router Benchmark Matrix

Measured on **Apple M4 (10-Core GPU, 16GB UMA)** using `bench_universal_router` with mandatory 32MB SLC cache flushing and double-precision CPU verification.

### 8B Transformer Tier ($K=4096, N=4096$)

| Prompt ($M$) | Format | Bits/Wt | GPU Latency | TFLOPS | Memory Bandwidth | tok/s | tok/s per BPW | Speedup vs Q4_0 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **33** | Q4_0 | 4.50 | 0.930 ms | 1.19 | 10.7 GB/s | 35,472 | 7,882.7 | 1.00x |
| | MLX 4-bit | 5.00 | 1.336 ms | 0.83 | 8.3 GB/s | 24,706 | 4,941.3 | 0.70x |
| | Q4_K | 4.50 | 1.117 ms | 0.99 | 8.9 GB/s | 29,557 | 6,568.1 | 0.83x |
| | **Ternary MMA** | **3.00** | **0.849 ms** | **1.30** | **8.1 GB/s** | **38,889** | **12,963.1** | **1.10x** |
| | Var-Rate Affine | 5.00 | 1.637 ms | 0.68 | 7.1 GB/s | 20,160 | 4,032.1 | 0.57x |
| | EXL3 Codebook | 4.50 | 1.272 ms | 0.87 | 7.8 GB/s | 25,949 | 5,766.4 | 0.73x |
| **128** | Q4_0 | 4.50 | 2.711 ms | 1.58 | 4.3 GB/s | 47,210 | 10,491.0 | 1.00x |
| | MLX 4-bit | 5.00 | 1.840 ms | 2.33 | 6.8 GB/s | 69,550 | 13,910.1 | 1.47x |
| | Q4_K | 4.50 | 2.280 ms | 1.88 | 5.1 GB/s | 56,143 | 12,476.3 | 1.19x |
| | **Ternary MMA** | **3.00** | **1.481 ms** | **2.90** | **5.7 GB/s** | **86,456** | **28,818.7** | **1.83x** |
| | Var-Rate Affine | 5.00 | 2.205 ms | 1.95 | 5.9 GB/s | 58,043 | 11,608.6 | 1.23x |
| | EXL3 Codebook | 4.50 | 2.326 ms | 1.85 | 5.0 GB/s | 55,035 | 12,229.9 | 1.17x |
| **2048** | Q4_0 | 4.50 | 22.009 ms | 3.12 | 2.0 GB/s | 93,052 | 20,678.2 | 1.00x |
| | **MLX 4-bit** | **5.00** | **21.744 ms** | **3.16** | **2.0 GB/s** | **94,185** | **18,837.0** | **1.01x** |
| | Q4_K | 4.50 | 24.622 ms | 2.79 | 1.7 GB/s | 83,177 | 18,483.9 | 0.89x |
| | Ternary MMA | 3.00 | 21.960 ms | 3.13 | 1.8 GB/s | 93,261 | 31,086.9 | 1.00x |
| | Var-Rate Affine | 5.00 | 26.426 ms | 2.60 | 1.7 GB/s | 77,498 | 15,499.6 | 0.83x |
| | EXL3 Codebook | 4.50 | 28.276 ms | 2.43 | 1.5 GB/s | 72,429 | 16,095.4 | 0.78x |
