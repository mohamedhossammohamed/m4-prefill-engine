"""
Red-Teaming Test Suite for Pluggable M4 vs. MLX Module Pair.

Performs adversarial stress-testing, vulnerability auditing, and invariant probing
against:
  - src/engine/modules.py
  - src/engine/__init__.py

Audit Vectors (13 Probes):
  Probe 1: Dimension Divisibility & Bounds Validation in M4QuantizedLinear.__init__
  Probe 2: Input Validation in compute_quantized_weight_bytes
  Probe 3: Arbitrary Leading Rank Tensor Dispatch (*batch_dims, K) including 4D
  Probe 4: Empty Tensor Handling [0, K], [B, 0, K], [B, H, 0, K]
  Probe 5: KV Cache Configured head_dim Validation
  Probe 6: KV Cache Keys vs. Values Shape Parity & Rank Invariant
  Probe 7: KV Cache Out-of-Core NVMe Direct I/O (fcntl F_NOCACHE)
  Probe 8: KV Cache Out-of-Core Retrieval & Streaming API
  Probe 9: dequantize_to_fp16_matrix Rejection of QUANT_VAR_RATE_AFFINE
  Probe 10: MLXQuantizedLinear Rejection of QUANT_VAR_RATE_AFFINE
  Probe 11: Format Parity & MLX Rejection of QUANT_EXL3 Super-Block
  Probe 12: Numerical Tripwires in M4QuantizedLinear (NaN/Inf Gating)
  Probe 13: Numerical Tripwires & Cache Poisoning in M4KVCache (NaN/Inf Gating)
"""

from __future__ import annotations
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np

from core.bridge.m4_bridge import (
    MetalUMABridge,
    m4_bridge_get_uma_footprint_mb,
    m4_bridge_init,
    m4_bridge_synchronize,
)
from src.engine import (
    FORMAT_SPECS,
    M4KVCache,
    M4QuantizedLinear,
    MLXKVCache,
    MLXQuantizedLinear,
    QuantFormat,
    compute_quantized_weight_bytes,
    create_kv_cache,
    create_linear,
    dequantize_to_fp16_matrix,
    generate_synthetic_quantized_weights,
    normalize_format,
)

# Terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class RedTeamAuditReport:
    """Telemetry collector and reporter for red-team vulnerabilities."""

    def __init__(self):
        self.findings: list[dict[str, str]] = []
        self.probes_run = 0
        self.probes_passed = 0
        self.probes_vulnerable = 0

    def record_finding(self, severity: str, category: str, summary: str, details: str):
        self.findings.append({
            "severity": severity,
            "category": category,
            "summary": summary,
            "details": details,
        })
        if severity in ("CRITICAL", "HIGH"):
            self.probes_vulnerable += 1

    def print_summary(self) -> int:
        print("\n" + "=" * 80)
        print(f"{BOLD}{CYAN}STEP 2 RED-TEAM AUDIT REPORT: PLUGGABLE M4 VS. MLX MODULES{RESET}")
        print("=" * 80)
        print(f"Total Audit Probes Run:     {self.probes_run}")
        print(f"Passed / Controlled:        {GREEN}{self.probes_passed}{RESET}")
        print(f"Vulnerabilities Identified: {RED if len(self.findings) > 0 else GREEN}{len(self.findings)}{RESET}")
        print("-" * 80)

        for i, f in enumerate(self.findings, 1):
            color = RED if f["severity"] in ("CRITICAL", "HIGH") else YELLOW
            print(f"{BOLD}[Finding {i}] [{color}{f['severity']}{RESET}{BOLD}] [{f['category']}]{RESET}")
            print(f"  Summary: {f['summary']}")
            print(f"  Details: {f['details']}\n")

        if len(self.findings) == 0:
            print(f"{BOLD}{GREEN}[✓] ALL 13 RED-TEAM AUDIT PROBES PASSED! ZERO VULNERABILITIES DETECTED.{RESET}")
        print("=" * 80)
        return len(self.findings)


REPORT = RedTeamAuditReport()


def log_probe(num: int, title: str):
    print(f"\n{BOLD}{CYAN}>>> [PROBE {num}/13] {title}{RESET}")


# ============================================================================
# 1. SHAPE & TENSOR DIMENSION INVARIANTS (PROBES 1 - 4)
# ============================================================================

def probe_01_m4_linear_init_bounds():
    """Probe 1 (Finding 1): Dimension Divisibility & Non-Positive Bounds in M4QuantizedLinear.__init__."""
    log_probe(1, "M4QuantizedLinear.__init__ Non-Positive Dimensions & Divisibility")
    REPORT.probes_run += 1

    # 1.1 Non-divisible in_features for standard block (Q4_0, block=32)
    try:
        M4QuantizedLinear(in_features=33, out_features=64, format=QuantFormat.QUANT_Q4_0)
        REPORT.record_finding("HIGH", "Shape Invariant", "Accepted in_features not divisible by block size", "in_features=33 was accepted.")
        return
    except ValueError:
        pass

    # 1.2 in_features = 0
    try:
        M4QuantizedLinear(in_features=0, out_features=64, format=QuantFormat.QUANT_Q4_0)
        REPORT.record_finding("HIGH", "Shape Invariant", "Accepted in_features=0", "in_features=0 was accepted.")
        return
    except ValueError:
        pass

    # 1.3 Negative in_features
    try:
        M4QuantizedLinear(in_features=-64, out_features=64, format=QuantFormat.QUANT_Q4_0)
        REPORT.record_finding("HIGH", "Shape Invariant", "Accepted negative in_features", "in_features=-64 was accepted.")
        return
    except ValueError:
        pass

    # 1.4 Non-positive out_features
    try:
        M4QuantizedLinear(in_features=64, out_features=0, format=QuantFormat.QUANT_Q4_0)
        REPORT.record_finding("HIGH", "Shape Invariant", "Accepted out_features=0", "out_features=0 was accepted.")
        return
    except ValueError:
        pass

    try:
        M4QuantizedLinear(in_features=64, out_features=-16, format=QuantFormat.QUANT_Q4_0)
        REPORT.record_finding("HIGH", "Shape Invariant", "Accepted out_features=-16", "out_features=-16 was accepted.")
        return
    except ValueError:
        pass

    # 1.5 Super-block divisibility by 256
    for sf in [QuantFormat.QUANT_Q4_K, QuantFormat.QUANT_VAR_RATE_AFFINE, QuantFormat.QUANT_EXL3]:
        try:
            M4QuantizedLinear(in_features=128, out_features=64, format=sf)
            REPORT.record_finding("HIGH", "Shape Invariant", f"Super-block {sf.name} accepted in_features=128", "Requires % 256 == 0.")
            return
        except ValueError:
            pass

    print("  [Pass] Non-positive features and divisibility constraints strictly enforced in M4QuantizedLinear.__init__.")
    REPORT.probes_passed += 1


def probe_02_weight_bytes_dimension_bounds():
    """Probe 2 (Finding 2): Non-positive dimensions validation in compute_quantized_weight_bytes."""
    log_probe(2, "compute_quantized_weight_bytes Input Validation")
    REPORT.probes_run += 1

    try:
        compute_quantized_weight_bytes(QuantFormat.QUANT_Q4_0, 0, 64)
        REPORT.record_finding("MEDIUM", "Input Validation", "compute_quantized_weight_bytes accepted in_features=0", "Failed to raise ValueError.")
        return
    except ValueError:
        pass

    try:
        compute_quantized_weight_bytes(QuantFormat.QUANT_Q4_0, -32, 64)
        REPORT.record_finding("MEDIUM", "Input Validation", "compute_quantized_weight_bytes accepted negative in_features", "Failed to raise ValueError.")
        return
    except ValueError:
        pass

    try:
        compute_quantized_weight_bytes(QuantFormat.QUANT_Q4_0, 64, 0)
        REPORT.record_finding("MEDIUM", "Input Validation", "compute_quantized_weight_bytes accepted out_features=0", "Failed to raise ValueError.")
        return
    except ValueError:
        pass

    # Valid check
    expected = (64 // 32) * 64 * 18
    actual = compute_quantized_weight_bytes(QuantFormat.QUANT_Q4_0, 64, 64)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print(f"  [Pass] compute_quantized_weight_bytes rejects non-positive dimensions and matches spec ({actual} bytes).")
    REPORT.probes_passed += 1


def probe_03_m4_linear_arbitrary_ranks_and_4d():
    """Probe 3 (Finding 3): Support arbitrary leading batch dimensions (*batch_dims, K) including 4D."""
    log_probe(3, "M4QuantizedLinear Generalized Leading Ranks & 4D Attention Tensors")
    REPORT.probes_run += 1

    K, N = 64, 64
    layer = M4QuantizedLinear(in_features=K, out_features=N, format=QuantFormat.QUANT_Q4_0)
    layer.init_synthetic_weights(scale=1.0)

    # 1. 1D input [K]
    x_1d = mx.ones((K,), dtype=mx.float16)
    y_1d = layer(x_1d)
    assert y_1d.shape == (N,), f"Expected ({N},), got {y_1d.shape}"

    # 2. 2D inputs [1, K] and [4, K]
    x_2d_1 = mx.ones((1, K), dtype=mx.float16)
    y_2d_1 = layer(x_2d_1)
    assert y_2d_1.shape == (1, N)

    x_2d_m = mx.ones((4, K), dtype=mx.float16)
    y_2d_m = layer(x_2d_m)
    assert y_2d_m.shape == (4, N)

    # 3. 3D input [B, M, K]
    x_3d = mx.ones((2, 4, K), dtype=mx.float16)
    y_3d = layer(x_3d)
    assert y_3d.shape == (2, 4, N)

    # 4. 4D input [B, H, M, K] (Standard Attention Projection Tensor)
    x_4d = mx.ones((2, 4, 8, K), dtype=mx.float16)
    try:
        y_4d = layer(x_4d)
        assert y_4d.shape == (2, 4, 8, N), f"Expected (2, 4, 8, {N}), got {y_4d.shape}"
    except Exception as e:
        REPORT.record_finding("MEDIUM", "Rank Invariant", "M4QuantizedLinear failed on 4D tensor", str(e))
        return

    # 5. 5D input [2, 3, 2, 4, K]
    x_5d = mx.ones((2, 3, 2, 4, K), dtype=mx.float16)
    y_5d = layer(x_5d)
    assert y_5d.shape == (2, 3, 2, 4, N), f"Expected (2, 3, 2, 4, {N}), got {y_5d.shape}"

    print(f"  [Pass] Generalized input ranks (1D, 2D, 3D, 4D {y_4d.shape}, 5D {y_5d.shape}) cleanly supported.")
    REPORT.probes_passed += 1


def probe_04_m4_linear_empty_tensors():
    """Probe 4 (Finding 4): Empty tensor handling [0, K], [B, 0, K], [B, H, 0, K]."""
    log_probe(4, "M4QuantizedLinear Empty Tensor Graceful Handling")
    REPORT.probes_run += 1

    K, N = 64, 64
    layer = M4QuantizedLinear(in_features=K, out_features=N, format=QuantFormat.QUANT_Q4_0)

    # 1. Empty 2D [0, K]
    x_empty_2d = mx.zeros((0, K), dtype=mx.float16)
    try:
        y_empty_2d = layer(x_empty_2d)
        assert y_empty_2d.shape == (0, N), f"Expected (0, {N}), got {y_empty_2d.shape}"
    except Exception as e:
        REPORT.record_finding("HIGH", "Empty Tensor", "M4QuantizedLinear crashed on 2D empty tensor [0, K]", str(e))
        return

    # 2. Empty 3D [2, 0, K]
    x_empty_3d = mx.zeros((2, 0, K), dtype=mx.float16)
    try:
        y_empty_3d = layer(x_empty_3d)
        assert y_empty_3d.shape == (2, 0, N), f"Expected (2, 0, {N}), got {y_empty_3d.shape}"
    except Exception as e:
        REPORT.record_finding("HIGH", "Empty Tensor", "M4QuantizedLinear crashed on 3D empty tensor [2, 0, K]", str(e))
        return

    # 3. Empty 4D [2, 4, 0, K]
    x_empty_4d = mx.zeros((2, 4, 0, K), dtype=mx.float16)
    try:
        y_empty_4d = layer(x_empty_4d)
        assert y_empty_4d.shape == (2, 4, 0, N), f"Expected (2, 4, 0, {N}), got {y_empty_4d.shape}"
    except Exception as e:
        REPORT.record_finding("HIGH", "Empty Tensor", "M4QuantizedLinear crashed on 4D empty tensor [2, 4, 0, K]", str(e))
        return

    print("  [Pass] 0-length tokens in dynamic batching return empty tensors without buffer crash.")
    REPORT.probes_passed += 1


# ============================================================================
# 2. MEMORY & KV CACHE ROBUSTNESS (PROBES 5 - 8)
# ============================================================================

def probe_05_kv_cache_head_dim_validation():
    """Probe 5 (Finding 5): Configured head_dim validation in M4KVCache.update_and_fetch."""
    log_probe(5, "M4KVCache Configured head_dim Invariant")
    REPORT.probes_run += 1

    H, D = 4, 64
    cache = M4KVCache(head_dim=D, n_heads=H, max_seq_len=64)

    # Mismatched head dimension D=32 vs configured D=64
    k_bad_d = mx.ones((1, H, 4, 32), dtype=mx.float16)
    v_bad_d = mx.ones((1, H, 4, 32), dtype=mx.float16)
    try:
        cache.update_and_fetch(k_bad_d, v_bad_d)
        REPORT.record_finding("HIGH", "Shape Invariant", "M4KVCache accepted mismatched head_dim", "keys head_dim=32 accepted for cache head_dim=64.")
        return
    except ValueError:
        pass

    # Valid head dimension
    k_ok = mx.ones((1, H, 4, D), dtype=mx.float16)
    v_ok = mx.ones((1, H, 4, D), dtype=mx.float16)
    k_out, v_out = cache.update_and_fetch(k_ok, v_ok)
    assert k_out.shape == (1, H, 4, D)

    print("  [Pass] Mismatched head_dim strictly rejected with ValueError.")
    REPORT.probes_passed += 1


def probe_06_kv_cache_shape_parity_and_rank():
    """Probe 6 (Finding 6): Keys vs. values shape parity and rank validation in M4KVCache."""
    log_probe(6, "M4KVCache Keys vs. Values Shape Parity & Rank Validation")
    REPORT.probes_run += 1

    H, D = 4, 64
    cache = M4KVCache(head_dim=D, n_heads=H, max_seq_len=64)

    # 1. 4D Keys vs 3D Values (Silent broadcasting prevention)
    k_4d = mx.ones((1, H, 2, D), dtype=mx.float16)
    v_3d = mx.ones((1, 2, D), dtype=mx.float16)
    try:
        cache.update_and_fetch(k_4d, v_3d)
        REPORT.record_finding("HIGH", "Shape Invariant", "M4KVCache accepted mismatched shapes (4D keys vs 3D values)", "Allowed silent broadcasting.")
        return
    except ValueError:
        pass

    # 2. Sequence length mismatch
    cache.reset()
    k_s2 = mx.ones((1, H, 2, D), dtype=mx.float16)
    v_s4 = mx.ones((1, H, 4, D), dtype=mx.float16)
    try:
        cache.update_and_fetch(k_s2, v_s4)
        REPORT.record_finding("CRITICAL", "Shape Invariant", "M4KVCache accepted mismatched sequence lengths", "k seq=2 vs v seq=4.")
        return
    except ValueError:
        pass

    # 3. Invalid rank (2D)
    cache.reset()
    k_2d = mx.ones((2, D), dtype=mx.float16)
    v_2d = mx.ones((2, D), dtype=mx.float16)
    try:
        cache.update_and_fetch(k_2d, v_2d)
        REPORT.record_finding("HIGH", "Shape Invariant", "M4KVCache accepted 2D tensor", "Only 3D/4D supported.")
        return
    except ValueError:
        pass

    print("  [Pass] Keys vs values shape parity and rank bounds enforced.")
    REPORT.probes_passed += 1


def probe_07_kv_cache_out_of_core_direct_io():
    """Probe 7 (Finding 7): Out-of-Core NVMe direct I/O configuration (fcntl F_NOCACHE)."""
    log_probe(7, "M4KVCache Out-of-Core Direct I/O (fcntl F_NOCACHE)")
    REPORT.probes_run += 1

    H, D = 4, 64
    with M4KVCache(head_dim=D, n_heads=H, mode="out_of_core", ram_capacity=8) as cache:
        if not cache.has_direct_io:
            REPORT.record_finding("HIGH", "Direct I/O Violation", "F_NOCACHE direct I/O not active on NVMe temporary file", "Direct I/O flag is False.")
            return

    print("  [Pass] NVMe temporary backing store configured with direct I/O (F_NOCACHE).")
    REPORT.probes_passed += 1


def probe_08_kv_cache_out_of_core_read_api():
    """Probe 8 (Finding 8): Out-of-Core NVMe retrieval API and streaming methods."""
    log_probe(8, "M4KVCache Out-of-Core NVMe Retrieval & Streaming API")
    REPORT.probes_run += 1

    H, D = 4, 64
    with M4KVCache(head_dim=D, n_heads=H, mode="out_of_core", ram_capacity=8) as cache:
        has_read = any(hasattr(cache, m) for m in ("read_stream", "fetch_out_of_core", "read_all_tokens"))
        if not has_read:
            REPORT.record_finding("HIGH", "Architectural Incompleteness", "M4KVCache lacks out-of-core retrieval method", "No read_stream / fetch_out_of_core found.")
            return

        # Write tokens and retrieve
        k = mx.ones((1, H, 12, D), dtype=mx.float16)
        v = mx.ones((1, H, 12, D), dtype=mx.float16)
        cache.update_and_fetch(k, v)

        k_tokens, v_tokens = cache.read_all_tokens()
        assert k_tokens is not None and v_tokens is not None
        assert k_tokens.shape == (1, H, 8, D) # Active RAM window

    print("  [Pass] Out-of-core retrieval API (read_stream, fetch_out_of_core, read_all_tokens) verified.")
    REPORT.probes_passed += 1


# ============================================================================
# 3. FORMAT PARITY ACROSS 6 CODECS (PROBES 9 - 11)
# ============================================================================

def probe_09_dequantize_var_rate_affine_rejection():
    """Probe 9 (Finding 9): dequantize_to_fp16_matrix rejection of QUANT_VAR_RATE_AFFINE."""
    log_probe(9, "dequantize_to_fp16_matrix Rejection of QUANT_VAR_RATE_AFFINE")
    REPORT.probes_run += 1

    K, N = 256, 64
    w_raw = generate_synthetic_quantized_weights(QuantFormat.QUANT_VAR_RATE_AFFINE, K, N, scale=1.0, seed=42)

    try:
        dequantize_to_fp16_matrix(w_raw, QuantFormat.QUANT_VAR_RATE_AFFINE, K, N)
        REPORT.record_finding("HIGH", "Format Parity", "dequantize_to_fp16_matrix returned matrix for QUANT_VAR_RATE_AFFINE", "Expected NotImplementedError.")
        return
    except NotImplementedError:
        pass

    print("  [Pass] dequantize_to_fp16_matrix cleanly raises NotImplementedError for QUANT_VAR_RATE_AFFINE.")
    REPORT.probes_passed += 1


def probe_10_mlx_var_rate_affine_rejection():
    """Probe 10 (Finding 10): MLX baseline rejection of QUANT_VAR_RATE_AFFINE preventing divergence."""
    log_probe(10, "MLXQuantizedLinear Rejection of QUANT_VAR_RATE_AFFINE")
    REPORT.probes_run += 1

    K, N = 256, 64
    w_raw = generate_synthetic_quantized_weights(QuantFormat.QUANT_VAR_RATE_AFFINE, K, N, scale=1.0, seed=42)

    try:
        MLXQuantizedLinear.from_quantized(w_raw, format=QuantFormat.QUANT_VAR_RATE_AFFINE, in_features=K, out_features=N)
        REPORT.record_finding("CRITICAL", "Numerical Discrepancy", "MLX baseline accepted QUANT_VAR_RATE_AFFINE without implementation", "Expected NotImplementedError.")
        return
    except NotImplementedError:
        pass

    print("  [Pass] MLXQuantizedLinear cleanly raises NotImplementedError, preventing silent benchmark corruption.")
    REPORT.probes_passed += 1


def probe_11_exl3_rejection_parity():
    """Probe 11 (Finding 11): Parity & MLX baseline rejection of QUANT_EXL3 super-block."""
    log_probe(11, "Super-Block QUANT_EXL3 Parity & MLX Baseline Rejection")
    REPORT.probes_run += 1

    K, N = 256, 64
    w_raw = generate_synthetic_quantized_weights(QuantFormat.QUANT_EXL3, K, N, scale=1.0, seed=42)

    # 1. dequantize_to_fp16_matrix must raise NotImplementedError
    try:
        dequantize_to_fp16_matrix(w_raw, QuantFormat.QUANT_EXL3, K, N)
        REPORT.record_finding("HIGH", "Format Parity", "dequantize_to_fp16_matrix allowed QUANT_EXL3", "Expected NotImplementedError.")
        return
    except NotImplementedError:
        pass

    # 2. MLX baseline must raise NotImplementedError
    try:
        MLXQuantizedLinear.from_quantized(w_raw, format=QuantFormat.QUANT_EXL3, in_features=K, out_features=N)
        REPORT.record_finding("HIGH", "Format Parity", "MLX baseline allowed QUANT_EXL3", "Expected NotImplementedError.")
        return
    except NotImplementedError:
        pass

    # 3. M4 Hardware kernel must execute cleanly
    m4_layer = M4QuantizedLinear.from_quantized(w_raw, format=QuantFormat.QUANT_EXL3, in_features=K, out_features=N)
    x = mx.ones((4, K), dtype=mx.float16)
    y_m4 = m4_layer(x)
    assert y_m4.shape == (4, N)
    assert bool(mx.all(mx.isfinite(y_m4)).item()) is True

    print("  [Pass] QUANT_EXL3 MLX emulation rejected while M4 hardware kernel executes cleanly.")
    REPORT.probes_passed += 1


# ============================================================================
# 4. TRIPWIRES & NUMERICAL INVARIANTS (PROBES 12 - 13)
# ============================================================================

def probe_12_linear_numerical_tripwires():
    """Probe 12 (Finding 12): M4QuantizedLinear tripwire gating on NaN/Inf activations."""
    log_probe(12, "M4QuantizedLinear Numerical Tripwires (check_finite=True)")
    REPORT.probes_run += 1

    K, N = 64, 64
    layer = M4QuantizedLinear(in_features=K, out_features=N, format=QuantFormat.QUANT_Q4_0)

    # 1. NaN activation with check_finite=True
    x_nan = mx.ones((2, K), dtype=mx.float16)
    x_nan[0, 5] = float("nan")
    try:
        layer(x_nan, check_finite=True)
        REPORT.record_finding("HIGH", "Tripwire Failure", "M4QuantizedLinear silently passed NaN with check_finite=True", "Expected FloatingPointError.")
        return
    except FloatingPointError:
        pass

    # 2. Inf activation with check_finite=True
    x_inf = mx.ones((2, K), dtype=mx.float16)
    x_inf[0, 5] = float("inf")
    try:
        layer(x_inf, check_finite=True)
        REPORT.record_finding("HIGH", "Tripwire Failure", "M4QuantizedLinear silently passed Inf with check_finite=True", "Expected FloatingPointError.")
        return
    except FloatingPointError:
        pass

    # 3. Clean activation with check_finite=True
    x_clean = mx.ones((2, K), dtype=mx.float16)
    y_clean = layer(x_clean, check_finite=True)
    assert y_clean.shape == (2, N)

    print("  [Pass] M4QuantizedLinear FloatingPointError tripwire triggered on NaN and Inf inputs.")
    REPORT.probes_passed += 1


def probe_13_kv_cache_numerical_tripwires():
    """Probe 13 (Finding 13): M4KVCache tripwire gating against NaN/Inf context poisoning."""
    log_probe(13, "M4KVCache Context Poisoning Tripwires (check_finite=True)")
    REPORT.probes_run += 1

    H, D = 4, 64
    cache = M4KVCache(head_dim=D, n_heads=H, max_seq_len=64)

    # 1. NaN injection
    k_nan = mx.ones((1, H, 1, D), dtype=mx.float16)
    k_nan[0, 0, 0, 0] = float("nan")
    v_clean = mx.ones((1, H, 1, D), dtype=mx.float16)
    try:
        cache.update_and_fetch(k_nan, v_clean, check_finite=True)
        REPORT.record_finding("HIGH", "Cache Poisoning", "M4KVCache accepted NaN with check_finite=True", "Expected FloatingPointError.")
        return
    except FloatingPointError:
        pass

    # 2. Inf injection
    k_inf = mx.ones((1, H, 1, D), dtype=mx.float16)
    k_inf[0, 0, 0, 0] = float("inf")
    try:
        cache.update_and_fetch(k_inf, v_clean, check_finite=True)
        REPORT.record_finding("HIGH", "Cache Poisoning", "M4KVCache accepted Inf with check_finite=True", "Expected FloatingPointError.")
        return
    except FloatingPointError:
        pass

    # 3. Clean insertion
    cache.reset()
    k_clean = mx.ones((1, H, 1, D), dtype=mx.float16)
    k_out, v_out = cache.update_and_fetch(k_clean, v_clean, check_finite=True)
    assert k_out.shape == (1, H, 1, D)

    print("  [Pass] M4KVCache FloatingPointError tripwire prevented persistent cache poisoning.")
    REPORT.probes_passed += 1


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> int:
    print(f"{BOLD}STARTING STEP 2 RED-TEAM HARDENING AUDIT: 13 VERIFICATION PROBES{RESET}")
    print("Target Modules: src/engine/modules.py, src/engine/__init__.py\n")

    m4_bridge_init()

    probe_01_m4_linear_init_bounds()
    probe_02_weight_bytes_dimension_bounds()
    probe_03_m4_linear_arbitrary_ranks_and_4d()
    probe_04_m4_linear_empty_tensors()
    probe_05_kv_cache_head_dim_validation()
    probe_06_kv_cache_shape_parity_and_rank()
    probe_07_kv_cache_out_of_core_direct_io()
    probe_08_kv_cache_out_of_core_read_api()
    probe_09_dequantize_var_rate_affine_rejection()
    probe_10_mlx_var_rate_affine_rejection()
    probe_11_exl3_rejection_parity()
    probe_12_linear_numerical_tripwires()
    probe_13_kv_cache_numerical_tripwires()

    num_findings = REPORT.print_summary()
    return 1 if num_findings > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
