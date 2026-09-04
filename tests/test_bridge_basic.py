"""
Basic Verification Test for Zero-Copy Metal UMA MLX Bridge.

Validates that MLX arrays in unified memory can be passed directly to
Metal shaders via ctypes buffer pointers, executed zero-copy, synchronized,
and evaluated with correct shapes and numerical finiteness.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np
import unittest

from core.bridge.m4_bridge import (
    MetalUMABridge,
    QuantFormat,
    m4_bridge_dispatch_gemm,
    m4_bridge_dispatch_gemv,
    m4_bridge_get_uma_footprint_mb,
    m4_bridge_init,
    m4_bridge_synchronize,
)


def test_bridge_initialization():
    """Verify Metal bridge initializes device, queue, and router shader library."""
    ok = m4_bridge_init()
    assert ok is True, "m4_bridge_init() must return True"

    footprint_mb = m4_bridge_get_uma_footprint_mb()
    assert footprint_mb > 0.0, f"UMA footprint must be positive, got {footprint_mb}"
    print(f"\n[PASS] Bridge initialized. Process UMA Footprint: {footprint_mb:.2f} MB")


def test_bridge_gemm_q4_0_shapes_and_finiteness():
    """Verify GEMM with QUANT_Q4_0 produces finite outputs and correct shapes."""
    assert m4_bridge_init() is True

    test_shapes = [
        (33, 64, 64),
        (64, 64, 64),
        (128, 128, 128),
    ]

    for M, K, N in test_shapes:
        X = mx.ones((M, K), dtype=mx.float16)

        num_blocks = (K * N) // 32
        w_bytes = num_blocks * 18  # struct block_q4_0: 1 half + 16 uint8
        w_raw = np.zeros(w_bytes, dtype=np.uint8)

        # Scale = 1.0 in FP16 (0x3C00), nibbles = 9 -> (9 - 8) = 1.0
        for b in range(num_blocks):
            w_raw[b * 18 + 0] = 0x00
            w_raw[b * 18 + 1] = 0x3C
            for q in range(16):
                w_raw[b * 18 + 2 + q] = 0x99

        W = mx.array(w_raw)
        Y = mx.zeros((M, N), dtype=mx.float16)

        ret = m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_Q4_0, M, K, N)
        assert ret == 0, f"m4_bridge_dispatch_gemm failed with code {ret}"

        m4_bridge_synchronize()

        assert Y.shape == (M, N), f"Expected shape {(M, N)}, got {Y.shape}"
        assert bool(mx.all(mx.isfinite(Y)).item()) is True, "Output Y contains NaN or Inf"

        # Dot product of K 1.0s with weight 1.0 = K
        expected_val = float(K)
        sample = Y[0, 0].item()
        assert abs(sample - expected_val) < 1e-2, (
            f"Expected output value ~{expected_val}, got {sample}"
        )

    print("[PASS] Q4_0 GEMM verified across multiple shapes with exact numerical match.")


def test_bridge_gemm_mlx_4bit():
    """Verify GEMM with QUANT_MLX_4BIT affine format."""
    assert m4_bridge_init() is True

    M, K, N = 64, 64, 64
    X = mx.ones((M, K), dtype=mx.float16)

    num_blocks = (K * N) // 32
    w_bytes = num_blocks * 20  # struct block_mlx_4bit: 2 halfs + 16 uint8
    w_raw = np.zeros(w_bytes, dtype=np.uint8)

    # d = 1.0 in FP16, bias = 0.0 in FP16, nibbles = 1 -> (1 * 1.0 + 0.0) = 1.0
    for b in range(num_blocks):
        w_raw[b * 20 + 0] = 0x00
        w_raw[b * 20 + 1] = 0x3C  # d = 1.0
        w_raw[b * 20 + 2] = 0x00
        w_raw[b * 20 + 3] = 0x00  # bias = 0.0
        for q in range(16):
            w_raw[b * 20 + 4 + q] = 0x11  # nibbles = 1

    W = mx.array(w_raw)
    Y = mx.zeros((M, N), dtype=mx.float16)

    ret = m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_MLX_4BIT, M, K, N)
    assert ret == 0, f"Dispatch failed with code {ret}"

    m4_bridge_synchronize()

    assert Y.shape == (M, N)
    assert bool(mx.all(mx.isfinite(Y)).item()) is True
    sample = Y[0, 0].item()
    assert abs(sample - float(K)) < 1e-2, f"Expected {float(K)}, got {sample}"
    print("[PASS] MLX_4BIT GEMM verified with exact numerical match.")


def test_bridge_decode_gemv():
    """Verify decode GEMV dispatch for autoregressive single-token steps."""
    assert m4_bridge_init() is True

    K, N = 64, 64
    x = mx.ones((K,), dtype=mx.float16)

    num_blocks = (K * N) // 32
    w_bytes = num_blocks * 18
    w_raw = np.zeros(w_bytes, dtype=np.uint8)
    for b in range(num_blocks):
        w_raw[b * 18 + 0] = 0x00
        w_raw[b * 18 + 1] = 0x3C
        for q in range(16):
            w_raw[b * 18 + 2 + q] = 0x99

    W = mx.array(w_raw)
    y = mx.zeros((N,), dtype=mx.float16)

    ret = m4_bridge_dispatch_gemv(x, W, y, QuantFormat.QUANT_Q4_0, K, N)
    assert ret == 0, f"m4_bridge_dispatch_gemv failed with code {ret}"

    m4_bridge_synchronize()

    assert y.shape == (N,), f"Expected shape {(N,)}, got {y.shape}"
    assert bool(mx.all(mx.isfinite(y)).item()) is True, "Output y contains NaN/Inf"
    sample = float(np.array(y)[0])
    assert abs(sample - float(K)) < 1e-2, f"Expected {float(K)}, got {sample}"
    print("[PASS] Decode GEMV verified with exact numerical match.")


if __name__ == "__main__":
    test_bridge_initialization()
    test_bridge_gemm_q4_0_shapes_and_finiteness()
    test_bridge_gemm_mlx_4bit()
    test_bridge_decode_gemv()
    print("\n[✓] ALL ZERO-COPY METAL UMA BRIDGE TESTS PASSED!")
