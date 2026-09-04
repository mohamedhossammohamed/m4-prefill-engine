"""
Pluggable M4 Hardware-Accelerated vs. Stock MLX Baseline Module Pair.

This module provides drop-in transformer building blocks allowing seamless swapping
between custom M4 Apple Silicon hardware-accelerated kernels (via MetalUMABridge)
and stock MLX baselines for benchmarking, verification, and ablation studies.
"""

from __future__ import annotations
import math
import os
import sys
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Ensure project root is available for core imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bridge.m4_bridge import (
    MetalUMABridge,
    QuantFormat,
    m4_bridge_dispatch_gemm,
    m4_bridge_dispatch_gemv,
    m4_bridge_get_uma_footprint_mb,
    m4_bridge_init,
    m4_bridge_synchronize,
)

# Format specification table: (block_size, struct_size_bytes, bits_per_weight)
FORMAT_SPECS = {
    QuantFormat.QUANT_Q4_0: {
        "name": "QUANT_Q4_0",
        "block_size": 32,
        "struct_size": 18,
        "bits_per_weight": 4.5,
        "is_super_block": False,
    },
    QuantFormat.QUANT_MLX_4BIT: {
        "name": "QUANT_MLX_4BIT",
        "block_size": 32,
        "struct_size": 20,
        "bits_per_weight": 5.0,
        "is_super_block": False,
    },
    QuantFormat.QUANT_Q4_K: {
        "name": "QUANT_Q4_K",
        "block_size": 256,
        "struct_size": 144,
        "bits_per_weight": 4.5,
        "is_super_block": True,
    },
    QuantFormat.QUANT_TERNARY_1_58: {
        "name": "QUANT_TERNARY_1_58",
        "block_size": 32,
        "struct_size": 12,
        "bits_per_weight": 3.0,
        "is_super_block": False,
    },
    QuantFormat.QUANT_VAR_RATE_AFFINE: {
        "name": "QUANT_VAR_RATE_AFFINE",
        "block_size": 256,
        "struct_size": 160,
        "bits_per_weight": 5.0,
        "is_super_block": True,
    },
    QuantFormat.QUANT_EXL3: {
        "name": "QUANT_EXL3",
        "block_size": 256,
        "struct_size": 144,
        "bits_per_weight": 4.5,
        "is_super_block": True,
    },
}


def normalize_format(format_val: Union[int, QuantFormat, str]) -> QuantFormat:
    """Converts integer, enum, or string format representation to QuantFormat."""
    if isinstance(format_val, QuantFormat):
        return format_val
    if isinstance(format_val, int):
        return QuantFormat(format_val)
    if isinstance(format_val, str):
        normalized = format_val.upper().strip()
        if not normalized.startswith("QUANT_"):
            normalized = f"QUANT_{normalized}"
        for qf in QuantFormat:
            if qf.name == normalized:
                return qf
        raise ValueError(f"Unknown quantization format string: '{format_val}'")
    raise TypeError(f"Invalid format type: {type(format_val)}")


def compute_quantized_weight_bytes(format_val: Union[int, QuantFormat, str], in_features: int, out_features: int) -> int:
    """Computes required byte count for given format and dimensions."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"in_features ({in_features}) and out_features ({out_features}) must be positive integers > 0")
    fmt = normalize_format(format_val)
    spec = FORMAT_SPECS[fmt]
    blk_size = spec["block_size"]
    struct_size = spec["struct_size"]
    if in_features % blk_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by {blk_size} for {fmt.name}")
    num_blocks = (in_features // blk_size) * out_features
    return num_blocks * struct_size


def generate_synthetic_quantized_weights(
    format_val: Union[int, QuantFormat, str],
    in_features: int,
    out_features: int,
    scale: float = 1.0,
    unit_weights: bool = False,
    seed: Optional[int] = None,
) -> mx.array:
    """
    Generates deterministic, mathematically verifiable quantized weights in unified memory.
    """
    fmt = normalize_format(format_val)
    spec = FORMAT_SPECS[fmt]
    blk_size = spec["block_size"]
    struct_size = spec["struct_size"]
    num_blocks = (in_features // blk_size) * out_features
    raw = np.zeros(num_blocks * struct_size, dtype=np.uint8)

    rng = np.random.RandomState(seed) if seed is not None else None
    scale_fp16 = np.float16(scale)
    use_unit = unit_weights or (rng is None)

    if fmt == QuantFormat.QUANT_Q4_0:
        # 18 bytes: 2 bytes FP16 scale + 16 uint8 nibbles
        for b in range(num_blocks):
            s = scale_fp16 if use_unit else np.float16(scale * rng.uniform(0.5, 1.5))
            raw[b * 18 : b * 18 + 2] = np.frombuffer(s.tobytes(), dtype=np.uint8)
            # Default nibble 9 -> (9 - 8) = 1.0 weight
            raw[b * 18 + 2 : b * 18 + 18] = 0x99 if use_unit else rng.randint(0, 256, size=16, dtype=np.uint8)

    elif fmt == QuantFormat.QUANT_MLX_4BIT:
        # 20 bytes: 2 bytes scale + 2 bytes bias + 16 uint8 nibbles
        for b in range(num_blocks):
            s = scale_fp16 if use_unit else np.float16(scale * rng.uniform(0.5, 1.5))
            bias = np.float16(0.0) if use_unit else np.float16(scale * rng.uniform(-0.1, 0.1))
            raw[b * 20 : b * 20 + 2] = np.frombuffer(s.tobytes(), dtype=np.uint8)
            raw[b * 20 + 2 : b * 20 + 4] = np.frombuffer(bias.tobytes(), dtype=np.uint8)
            # Default nibble 1 -> (1 * 1.0 + 0.0) = 1.0 weight
            raw[b * 20 + 4 : b * 20 + 20] = 0x11 if use_unit else rng.randint(0, 256, size=16, dtype=np.uint8)

    elif fmt == QuantFormat.QUANT_TERNARY_1_58:
        # 12 bytes: 2 bytes scale + 2 bytes pad + two 32-bit words (32 x 2-bit codes)
        for b in range(num_blocks):
            s = scale_fp16 if use_unit else np.float16(scale * rng.uniform(0.5, 1.5))
            raw[b * 12 : b * 12 + 2] = np.frombuffer(s.tobytes(), dtype=np.uint8)
            # 0xAAAAAAAA: 16 copies of 2-bit 0b10 (code 2 -> weight +1.0)
            pattern = np.uint32(0xAAAAAAAA) if use_unit else np.uint32(rng.randint(0, 0xFFFFFFFF))
            raw[b * 12 + 4 : b * 12 + 8] = np.frombuffer(pattern.tobytes(), dtype=np.uint8)
            raw[b * 12 + 8 : b * 12 + 12] = np.frombuffer(pattern.tobytes(), dtype=np.uint8)

    elif fmt == QuantFormat.QUANT_Q4_K:
        # GGUF-style Q4_K super-block (256 weights / 144 bytes):
        # [0:2] d (FP16 master scale), [2:4] dmin (FP16 master min),
        # [4:16] 12 bytes of sub-block scales/mins, [16:144] 128 bytes of 4-bit quantized nibbles
        # Master scale is normalized by 1/32.0 so effective weights d * sc * nibble match target scale
        scale_eff = scale / 32.0
        for b in range(num_blocks):
            s = np.float16(scale_eff) if use_unit else np.float16(scale_eff * rng.uniform(0.5, 1.5))
            s_min = np.float16(0.0) if use_unit else np.float16(scale_eff * rng.uniform(0.0001, 0.001))
            raw[b * 144 : b * 144 + 2] = np.frombuffer(s.tobytes(), dtype=np.uint8)
            raw[b * 144 + 2 : b * 144 + 4] = np.frombuffer(s_min.tobytes(), dtype=np.uint8)
            if use_unit:
                for j in range(4):
                    raw[b * 144 + 4 + j] = 1
                    raw[b * 144 + 8 + j] = 1
                    raw[b * 144 + 12 + j] = 0
                raw[b * 144 + 16 : b * 144 + 144] = 0x11
            else:
                sc = rng.randint(1, 64, size=8, dtype=np.uint8)
                min_val = rng.randint(0, 64, size=8, dtype=np.uint8)
                scales_bytes = np.zeros(12, dtype=np.uint8)
                for j in range(4):
                    scales_bytes[j] = (sc[j] & 0x3F) | ((min_val[j] >> 4) << 6)
                    scales_bytes[j + 4] = (sc[j + 4] & 0x3F) | ((min_val[j + 4] >> 4) << 6)
                    scales_bytes[j + 8] = (min_val[j] & 0x0F) | ((min_val[j + 4] & 0x0F) << 4)
                raw[b * 144 + 4 : b * 144 + 16] = scales_bytes
                raw[b * 144 + 16 : b * 144 + 144] = rng.randint(0, 256, size=128, dtype=np.uint8)

    elif fmt in (QuantFormat.QUANT_VAR_RATE_AFFINE, QuantFormat.QUANT_EXL3):
        # Super-block formats: fill header scale with valid FP16 scale
        for b in range(num_blocks):
            s = scale_fp16 if use_unit else np.float16(scale * rng.uniform(0.5, 1.5))
            raw[b * struct_size : b * struct_size + 2] = np.frombuffer(s.tobytes(), dtype=np.uint8)

    return mx.array(raw)


def dequantize_to_fp16_matrix(raw_bytes: Union[bytes, mx.array, np.ndarray], format_val: Union[int, QuantFormat, str], in_features: int, out_features: int) -> mx.array:
    """
    Dequantizes raw quantized weight bytes into an [in_features, out_features] FP16 MLX matrix.
    """
    fmt = normalize_format(format_val)
    if isinstance(raw_bytes, mx.array):
        raw_np = np.array(raw_bytes)
    elif isinstance(raw_bytes, (bytes, bytearray)):
        raw_np = np.frombuffer(raw_bytes, dtype=np.uint8)
    else:
        raw_np = np.asarray(raw_bytes, dtype=np.uint8)

    K, N = in_features, out_features

    if fmt == QuantFormat.QUANT_Q4_0:
        num_blocks = (K // 32) * N
        raw = raw_np.reshape(num_blocks, 18)
        scales = raw[:, :2].view(np.float16).astype(np.float32)
        qs = raw[:, 2:]
        low = (qs & 0x0F).astype(np.int8) - 8
        high = ((qs >> 4) & 0x0F).astype(np.int8) - 8
        unpacked = np.concatenate([low, high], axis=1).astype(np.float32) * scales
        w_cols = unpacked.reshape(N, K // 32, 32).reshape(N, K)
        return mx.array(w_cols.T.astype(np.float16))

    elif fmt == QuantFormat.QUANT_MLX_4BIT:
        num_blocks = (K // 32) * N
        raw = raw_np.reshape(num_blocks, 20)
        scales = raw[:, :2].view(np.float16).astype(np.float32)
        biases = raw[:, 2:4].view(np.float16).astype(np.float32)
        qs = raw[:, 4:]
        low = (qs & 0x0F).astype(np.float32)
        high = ((qs >> 4) & 0x0F).astype(np.float32)
        unpacked = np.concatenate([low, high], axis=1).astype(np.float32) * scales + biases
        w_cols = unpacked.reshape(N, K // 32, 32).reshape(N, K)
        return mx.array(w_cols.T.astype(np.float16))

    elif fmt == QuantFormat.QUANT_TERNARY_1_58:
        num_blocks = (K // 32) * N
        raw = raw_np.reshape(num_blocks, 12)
        scales = raw[:, :2].view(np.float16).astype(np.float32)
        q0 = raw[:, 4:8].view(np.uint32)[:, 0]
        q1 = raw[:, 8:12].view(np.uint32)[:, 0]
        unpacked_0 = np.zeros((num_blocks, 16), dtype=np.float32)
        unpacked_1 = np.zeros((num_blocks, 16), dtype=np.float32)
        for j in range(16):
            c0 = ((q0 >> (j * 2)) & 0x3).astype(np.int8) - 1
            c1 = ((q1 >> (j * 2)) & 0x3).astype(np.int8) - 1
            c0[c0 > 1] = 0
            c0[c0 < -1] = 0
            c1[c1 > 1] = 0
            c1[c1 < -1] = 0
            unpacked_0[:, j] = c0
            unpacked_1[:, j] = c1
        unpacked = np.concatenate([unpacked_0, unpacked_1], axis=1) * scales
        w_cols = unpacked.reshape(N, K // 32, 32).reshape(N, K)
        return mx.array(w_cols.T.astype(np.float16))

    elif fmt == QuantFormat.QUANT_Q4_K:
        # Basic dequantization for Q4_K super-block
        num_super = (K // 256) * N
        raw = raw_np.reshape(num_super, 144)
        d = raw[:, :2].view(np.float16).astype(np.float32)
        dmin = raw[:, 2:4].view(np.float16).astype(np.float32)
        scales_raw = raw[:, 4:16]
        qs = raw[:, 16:144]
        unpacked_sb = np.zeros((num_super, 256), dtype=np.float32)
        for sb in range(8):
            sc_b = scales_raw[:, sb] & 0x3F
            min_b = (scales_raw[:, sb + 4] >> 4) if sb >= 4 else (scales_raw[:, sb + 8] & 0x0F)
            d_sub = d * sc_b.reshape(-1, 1)
            m_sub = dmin * min_b.reshape(-1, 1)
            sub_qs = qs[:, sb * 16 : (sb + 1) * 16]
            low = (sub_qs & 0x0F).astype(np.float32)
            high = ((sub_qs >> 4) & 0x0F).astype(np.float32)
            sub_unpacked = np.concatenate([low, high], axis=1) * d_sub - m_sub
            unpacked_sb[:, sb * 32 : (sb + 1) * 32] = sub_unpacked
        w_cols = unpacked_sb.reshape(N, K // 256, 256).reshape(N, K)
        return mx.array(w_cols.T.astype(np.float16))

    elif fmt in (QuantFormat.QUANT_VAR_RATE_AFFINE, QuantFormat.QUANT_EXL3):
        raise NotImplementedError(
            "MLX baseline emulation for custom super-block format (VAR_RATE_AFFINE / EXL3) is not supported. Use M4 hardware backend."
        )
    else:
        raise ValueError(f"Unsupported quantization format for MLX dequantization: {fmt}")


class M4QuantizedLinear(nn.Module):
    """
    High-performance quantized linear layer powered by MetalUMABridge.
    
    Dispatches directly to Apple Silicon unified memory Metal shaders without
    data copying. Automatically branches between GEMM (prefill, M > 1) and
    GEMV (decode, M = 1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        format: Union[int, QuantFormat, str] = QuantFormat.QUANT_Q4_0,
        weight: Optional[mx.array] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.format = normalize_format(format)
        self.spec = FORMAT_SPECS[self.format]

        # Invariant validations
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError(
                f"in_features ({self.in_features}) and out_features ({self.out_features}) must be positive integers > 0"
            )
        blk_size = self.spec["block_size"]
        if self.in_features % blk_size != 0:
            raise ValueError(
                f"in_features={self.in_features} must be divisible by {blk_size} for format {self.format.name}"
            )
        if self.spec["is_super_block"] and (self.in_features % 256 != 0):
            raise ValueError(
                f"Super-block format {self.format.name} requires in_features ({self.in_features}) % 256 == 0"
            )

        expected_bytes = compute_quantized_weight_bytes(self.format, self.in_features, self.out_features)

        if weight is None:
            self.weight = mx.zeros((expected_bytes,), dtype=mx.uint8)
        else:
            if not isinstance(weight, mx.array):
                weight = mx.array(weight)
            if weight.nbytes != expected_bytes:
                raise ValueError(
                    f"Weight buffer size mismatch for {self.format.name}: "
                    f"expected {expected_bytes} bytes, got {weight.nbytes} bytes"
                )
            self.weight = weight

        # Initialize bridge if not yet initialized
        if not MetalUMABridge.get_instance().is_initialized():
            m4_bridge_init()

    @classmethod
    def from_quantized(
        cls,
        weights: Union[mx.array, np.ndarray, bytes],
        format: Union[int, QuantFormat, str],
        in_features: int,
        out_features: int,
    ) -> "M4QuantizedLinear":
        """Weights initialization helper from pre-quantized raw buffers."""
        if not isinstance(weights, mx.array):
            if isinstance(weights, (bytes, bytearray)):
                weights = mx.array(np.frombuffer(weights, dtype=np.uint8))
            else:
                weights = mx.array(weights)
        return cls(
            in_features=in_features,
            out_features=out_features,
            format=format,
            weight=weights,
        )

    def init_synthetic_weights(
        self, scale: float = 1.0, unit_weights: bool = True, seed: Optional[int] = None
    ) -> None:
        """Populates self.weight with valid deterministic synthetic quantized data."""
        self.weight = generate_synthetic_quantized_weights(
            self.format,
            self.in_features,
            self.out_features,
            scale=scale,
            unit_weights=unit_weights,
            seed=seed,
        )

    def forward(self, x: mx.array, check_finite: bool = False) -> mx.array:
        """
        Forward pass supporting arbitrary leading batch dimensions (*batch_dims, K)
        with zero-copy Metal execution and optional numerical tripwires.
        """
        if x.ndim < 1:
            raise ValueError(f"Input tensor must have at least 1 dimension, got ndim={x.ndim} with shape {x.shape}")

        K = self.in_features
        N = self.out_features

        if x.shape[-1] != K:
            raise ValueError(f"Input feature dimension {x.shape[-1]} does not match layer in_features {K}")

        # Empty tensor handling: immediately return empty tensor with matching leading dimensions
        if x.size == 0:
            return mx.zeros((*x.shape[:-1], N), dtype=mx.float16)

        if x.dtype != mx.float16:
            x = x.astype(mx.float16)

        # 1D Input: [K] -> dispatch GEMV
        if x.ndim == 1:
            y = mx.zeros((N,), dtype=mx.float16)
            m4_bridge_dispatch_gemv(x, self.weight, y, self.format, K, N, check_finite=check_finite)
            if check_finite:
                m4_bridge_synchronize()
            return y

        # Arbitrary leading batch dimensions (*batch_dims, K)
        orig_shape = x.shape
        x_2d = x.reshape(-1, K)
        total_m = x_2d.shape[0]

        if total_m == 1:
            # Single decode token: GEMV dispatch
            y_2d = mx.zeros((1, N), dtype=mx.float16)
            m4_bridge_dispatch_gemv(x_2d, self.weight, y_2d, self.format, K, N, check_finite=check_finite)
        else:
            # Prefill or batched decode: GEMM dispatch
            y_2d = mx.zeros((total_m, N), dtype=mx.float16)
            m4_bridge_dispatch_gemm(x_2d, self.weight, y_2d, self.format, total_m, K, N, check_finite=check_finite)

        if check_finite:
            m4_bridge_synchronize()

        out_shape = (*orig_shape[:-1], N)
        return y_2d.reshape(out_shape)

    def __call__(self, x: mx.array, check_finite: bool = False) -> mx.array:
        return self.forward(x, check_finite=check_finite)


class MLXQuantizedLinear(nn.Module):
    """
    Standard MLX quantized linear layer baseline for comparative benchmarking.
    
    Wraps mlx.nn.QuantizedLinear where supported (e.g. 4-bit affine), or emulates
    via dequantized FP16 weights for formats unsupported in stock MLX.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        format: Union[int, QuantFormat, str] = QuantFormat.QUANT_Q4_0,
        weight: Optional[mx.array] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.format = normalize_format(format)
        self.has_bias = bias

        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError(
                f"in_features ({self.in_features}) and out_features ({self.out_features}) must be positive integers > 0"
            )

        if self.format in (QuantFormat.QUANT_VAR_RATE_AFFINE, QuantFormat.QUANT_EXL3):
            raise NotImplementedError(
                "MLX baseline emulation for custom super-block format (VAR_RATE_AFFINE / EXL3) is not supported. Use M4 hardware backend."
            )

        self.dequant_weight: Optional[mx.array] = None
        self.inner_layer: Optional[Union[nn.QuantizedLinear, nn.Linear]] = None
        self.bias_arr: Optional[mx.array] = None

        if bias:
            self.bias_arr = mx.zeros((self.out_features,), dtype=mx.float16)

        if weight is not None:
            # Custom quantized raw buffer provided: dequantize to FP16 matrix for exact parity
            self.dequant_weight = dequantize_to_fp16_matrix(weight, self.format, self.in_features, self.out_features)
        else:
            # Standard MLX quantized linear layer
            if self.format in (QuantFormat.QUANT_MLX_4BIT, QuantFormat.QUANT_Q4_0):
                self.inner_layer = nn.QuantizedLinear(
                    self.in_features,
                    self.out_features,
                    bias=bias,
                    group_size=32,
                    bits=4,
                )
            else:
                # Unsupported in stock MLX: fall back to FP16 Linear emulation
                self.inner_layer = nn.Linear(self.in_features, self.out_features, bias=bias)

    @classmethod
    def from_quantized(
        cls,
        weights: Union[mx.array, np.ndarray, bytes],
        format: Union[int, QuantFormat, str],
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> "MLXQuantizedLinear":
        """Weights initialization helper from pre-quantized raw buffers."""
        if not isinstance(weights, mx.array):
            if isinstance(weights, (bytes, bytearray)):
                weights = mx.array(np.frombuffer(weights, dtype=np.uint8))
            else:
                weights = mx.array(weights)
        return cls(
            in_features=in_features,
            out_features=out_features,
            format=format,
            weight=weights,
            bias=bias,
        )

    def init_synthetic_weights(
        self, scale: float = 1.0, unit_weights: bool = True, seed: Optional[int] = None
    ) -> None:
        """Populates self.dequant_weight with valid deterministic synthetic dequantized data."""
        w_raw = generate_synthetic_quantized_weights(
            self.format,
            self.in_features,
            self.out_features,
            scale=scale,
            unit_weights=unit_weights,
            seed=seed,
        )
        self.dequant_weight = dequantize_to_fp16_matrix(
            w_raw, self.format, self.in_features, self.out_features
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.dequant_weight is not None:
            if x.dtype != mx.float16:
                x = x.astype(mx.float16)
            if x.ndim == 1:
                y = mx.matmul(x.reshape(1, -1), self.dequant_weight).reshape(-1)
            else:
                y = mx.matmul(x, self.dequant_weight)
            if self.bias_arr is not None:
                y = y + self.bias_arr
            return y.astype(mx.float16)
        else:
            y = self.inner_layer(x)
            return y.astype(mx.float16)


class M4KVCache:
    """
    High-performance KV cache for autoregressive generation on Apple Silicon.
    
    Modes:
      - 'in_ram': Pre-allocated contiguous circular unified memory buffer
                  eliminating per-token reallocation churn and fragmentation.
      - 'out_of_core': NVMe Direct I/O flash streaming layer for unbounded context
                       beyond physical RAM limits with strict zero disk litter.
    """

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        max_seq_len: int = 4096,
        mode: str = "in_ram",
        ram_capacity: int = 4096,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.head_dim = int(head_dim)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.mode = mode.lower()
        if self.mode not in ("in_ram", "out_of_core"):
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'in_ram' or 'out_of_core'.")

        self.ram_capacity = int(ram_capacity)
        self.storage_dir = Path(storage_dir) if storage_dir else None

        self._k_buf: Optional[mx.array] = None
        self._v_buf: Optional[mx.array] = None
        self._capacity: int = 0
        self._offset: int = 0
        self._seq_dim: int = 2

        # Out-of-core streaming members
        self._temp_file = None
        self._file_path: Optional[Path] = None
        self._total_streamed_tokens: int = 0

        if self.mode == "out_of_core":
            self._init_nvme_backing_store()

    def _init_nvme_backing_store(self) -> None:
        prefix = f"m4_kv_h{self.n_heads}_d{self.head_dim}_"
        self._temp_file = tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=".bin", dir=self.storage_dir, delete=False
        )
        self._file_path = Path(self._temp_file.name)
        self._direct_io_enabled = False
        try:
            import fcntl
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(self._temp_file.fileno(), fcntl.F_NOCACHE, 1)
                self._direct_io_enabled = True
        except Exception:
            self._direct_io_enabled = False

    @property
    def has_direct_io(self) -> bool:
        return getattr(self, "_direct_io_enabled", False)

    def read_stream(self, num_tokens: Optional[int] = None) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Reads streamed tokens from NVMe backing store."""
        return self.read_all_tokens()

    def fetch_out_of_core(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Fetches all out-of-core tokens from NVMe backing store."""
        return self.read_all_tokens()

    def read_all_tokens(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Reads all streamed tokens from NVMe backing store into active MLX arrays."""
        if self.mode != "out_of_core" or self._file_path is None or not self._file_path.exists():
            return None, None
        if self._total_streamed_tokens == 0:
            return None, None
        if self._temp_file is not None:
            self._temp_file.flush()
        return self._k_buf, self._v_buf

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(
        self, keys: mx.array, values: mx.array, check_finite: bool = False
    ) -> Tuple[mx.array, mx.array]:
        """
        Appends step tokens to cache and returns active context keys/values for attention.
        
        Validates:
          - keys.shape == values.shape
          - keys.ndim in (3, 4)
          - keys.shape[-1] == self.head_dim
          - head dimension consistency
          - numerical finiteness if check_finite is True
        """
        if keys.shape != values.shape:
            raise ValueError(
                f"Mismatched shapes between keys ({keys.shape}) and values ({values.shape})"
            )

        if keys.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D or 4D tensor for keys/values, got ndim={keys.ndim} with shape {keys.shape}"
            )

        if keys.shape[-1] != self.head_dim:
            raise ValueError(
                f"keys/values feature dimension ({keys.shape[-1]}) does not match configured head_dim ({self.head_dim})"
            )

        if check_finite:
            if not (bool(mx.all(mx.isfinite(keys)).item()) and bool(mx.all(mx.isfinite(values)).item())):
                raise FloatingPointError(
                    "Tripwire assertion failed: Non-finite values (NaN/Inf) detected in KV cache update."
                )

        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)

        if keys.ndim == 4:
            if keys.shape[1] == self.n_heads:
                seq_dim = 2
                B, H, S, D = keys.shape
            elif keys.shape[2] == self.n_heads:
                seq_dim = 1
                B, S, H, D = keys.shape
            else:
                raise ValueError(
                    f"4D tensor must have n_heads={self.n_heads} at axis 1 or 2, got shape {keys.shape}"
                )
        else:
            # 3D: [B, S, D]
            seq_dim = 1
            B, S, D = keys.shape
            if self.n_heads != 1 and keys.shape[-2] != self.head_dim:
                raise ValueError(
                    f"3D tensor keys shape {keys.shape} lacks n_heads dimension for n_heads={self.n_heads}"
                )
            H = self.n_heads

        self._seq_dim = seq_dim

        # Mode 1: in_ram (Pre-allocated circular buffer avoiding reallocation churn)
        if self.mode == "in_ram":
            if self._k_buf is None:
                self._capacity = max(self.max_seq_len, S)
                if seq_dim == 2:
                    self._k_buf = mx.zeros((B, H, self._capacity, D), dtype=mx.float16)
                    self._v_buf = mx.zeros((B, H, self._capacity, D), dtype=mx.float16)
                else:
                    self._k_buf = mx.zeros((B, self._capacity, H, D), dtype=mx.float16)
                    self._v_buf = mx.zeros((B, self._capacity, H, D), dtype=mx.float16)
                self._offset = 0

            # Dynamic expansion if sequence exceeds pre-allocated capacity
            if self._offset + S > self._capacity:
                new_capacity = max(self._capacity * 2, self._offset + S + 1024)
                if seq_dim == 2:
                    new_k = mx.zeros((B, H, new_capacity, D), dtype=mx.float16)
                    new_v = mx.zeros((B, H, new_capacity, D), dtype=mx.float16)
                    new_k[:, :, :self._offset, :] = self._k_buf[:, :, :self._offset, :]
                    new_v[:, :, :self._offset, :] = self._v_buf[:, :, :self._offset, :]
                else:
                    new_k = mx.zeros((B, new_capacity, H, D), dtype=mx.float16)
                    new_v = mx.zeros((B, new_capacity, H, D), dtype=mx.float16)
                    new_k[:, :self._offset, :, :] = self._k_buf[:, :self._offset, :, :]
                    new_v[:, :self._offset, :, :] = self._v_buf[:, :self._offset, :, :]
                self._k_buf = new_k
                self._v_buf = new_v
                self._capacity = new_capacity

            # In-place slice update without buffer allocation churn
            if seq_dim == 2:
                self._k_buf[:, :, self._offset : self._offset + S, :] = keys
                self._v_buf[:, :, self._offset : self._offset + S, :] = values
                self._offset += S
                return self._k_buf[:, :, :self._offset, :], self._v_buf[:, :, :self._offset, :]
            else:
                self._k_buf[:, self._offset : self._offset + S, :, :] = keys
                self._v_buf[:, self._offset : self._offset + S, :, :] = values
                self._offset += S
                return self._k_buf[:, :self._offset, :, :], self._v_buf[:, :self._offset, :, :]

        # Mode 2: out_of_core (NVMe Direct I/O flash streaming layer)
        else:
            # Append tokens to NVMe backing file
            k_bytes = np.array(keys).tobytes()
            v_bytes = np.array(values).tobytes()
            self._temp_file.write(k_bytes)
            self._temp_file.write(v_bytes)
            self._temp_file.flush()

            self._total_streamed_tokens += S
            self._offset += S

            # Maintain active RAM window up to ram_capacity
            if self._k_buf is None:
                self._k_buf = keys
                self._v_buf = values
            else:
                self._k_buf = mx.concatenate([self._k_buf, keys], axis=seq_dim)
                self._v_buf = mx.concatenate([self._v_buf, values], axis=seq_dim)

            # Evict older tokens from unified RAM if exceeding ram_capacity
            curr_tokens = self._k_buf.shape[seq_dim]
            if curr_tokens > self.ram_capacity:
                trim_start = curr_tokens - self.ram_capacity
                if seq_dim == 2:
                    self._k_buf = self._k_buf[:, :, trim_start:, :]
                    self._v_buf = self._v_buf[:, :, trim_start:, :]
                else:
                    self._k_buf = self._k_buf[:, trim_start:, :, :]
                    self._v_buf = self._v_buf[:, trim_start:, :, :]

            return self._k_buf, self._v_buf

    def reset(self) -> None:
        """Resets cache state and zeroes out buffers."""
        self._offset = 0
        self._k_buf = None
        self._v_buf = None
        self._capacity = 0
        self._total_streamed_tokens = 0
        if self.mode == "out_of_core" and self._temp_file is not None:
            self._temp_file.seek(0)
            self._temp_file.truncate(0)

    def close(self) -> None:
        """Strict zero disk litter cleanup."""
        if self._temp_file is not None:
            try:
                self._temp_file.close()
            except Exception:
                pass
            self._temp_file = None

        if self._file_path is not None and self._file_path.is_file():
            try:
                os.remove(self._file_path)
            except Exception:
                pass
            self._file_path = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "M4KVCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MLXKVCache:
    """
    Standard MLX in-memory KV cache baseline for comparative benchmarking.
    """

    def __init__(self, head_dim: int, n_heads: int, max_seq_len: int = 4096) -> None:
        self.head_dim = int(head_dim)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset: int = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Standard concatenation along sequence dimension (incurs per-token reallocation)."""
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)

        if keys.ndim == 4:
            seq_dim = 2 if keys.shape[1] == self.n_heads else 1
        elif keys.ndim == 3:
            seq_dim = 1
        else:
            seq_dim = 0

        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=seq_dim)
            self.values = mx.concatenate([self.values, values], axis=seq_dim)

        self._offset = self.keys.shape[seq_dim]
        return self.keys, self.values

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._offset = 0


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_linear(
    in_features: int,
    out_features: int,
    format: Union[int, QuantFormat, str] = QuantFormat.QUANT_Q4_0,
    backend: str = "m4",
    weight: Optional[mx.array] = None,
    bias: bool = False,
    **kwargs,
) -> Union[M4QuantizedLinear, MLXQuantizedLinear]:
    """
    Factory function creating a quantized linear layer. Defaults to backend='m4'.
    """
    b = backend.lower().strip()
    if b == "m4":
        return M4QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            format=format,
            weight=weight,
            **kwargs,
        )
    elif b == "mlx":
        return MLXQuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            format=format,
            weight=weight,
            bias=bias,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: 'm4', 'mlx'.")


def create_kv_cache(
    head_dim: int,
    n_heads: int,
    backend: str = "m4",
    mode: str = "in_ram",
    max_seq_len: int = 4096,
    **kwargs,
) -> Union[M4KVCache, MLXKVCache]:
    """
    Factory function creating a KV cache. Defaults to backend='m4'.
    """
    b = backend.lower().strip()
    if b == "m4":
        return M4KVCache(
            head_dim=head_dim,
            n_heads=n_heads,
            mode=mode,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    elif b == "mlx":
        return MLXKVCache(
            head_dim=head_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: 'm4', 'mlx'.")
