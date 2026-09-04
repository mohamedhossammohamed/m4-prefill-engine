"""
Red-Teaming Test Suite for Zero-Copy Metal UMA MLX Bridge.

Performs adversarial auditing, invariant stress-testing, and vulnerability probing
against the Metal UMA bridge implementation:
  - Memory & Alignment Invariants (strided MLX views, non-contiguous layouts, page alignment)
  - Format Coverage & Codec Robustness (all 6 formats, invalid format IDs, buffer bounds)
  - Edge Case Dimension Boundaries (M in {1, 33, 127, 128, 129, 2048}, N & K non-divisible by 64)
  - Garbage Collection & Race Conditions (asynchronous drop of in-flight arrays)
  - Metrology Compliance (task_vm_info.phys_footprint UMA tracking, NaN/Inf tripwires)
"""

from __future__ import annotations
import ctypes
import gc
import mmap
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np

from core.bridge.m4_bridge import (
    MetalUMABridge,
    QuantFormat,
    m4_bridge_dispatch_gemm,
    m4_bridge_dispatch_gemv,
    m4_bridge_get_uma_footprint_mb,
    m4_bridge_init,
    m4_bridge_synchronize,
)

# Color terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class RedTeamAuditReport:
    def __init__(self):
        self.findings: list[dict[str, str]] = []
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_vulnerable = 0

    def record_finding(self, severity: str, category: str, summary: str, details: str):
        self.findings.append({
            "severity": severity,
            "category": category,
            "summary": summary,
            "details": details,
        })
        if severity in ("HIGH", "CRITICAL"):
            self.tests_vulnerable += 1

    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"{BOLD}{CYAN}STEP 1 RED-TEAM AUDIT REPORT: ZERO-COPY METAL UMA BRIDGE{RESET}")
        print("=" * 80)
        print(f"Total Test Probes Run: {self.tests_run}")
        print(f"Passed / Controlled:   {GREEN}{self.tests_passed}{RESET}")
        print(f"Identified Vulnerabilities: {RED if self.tests_vulnerable > 0 else GREEN}{len(self.findings)}{RESET}")
        print("-" * 80)

        for i, f in enumerate(self.findings, 1):
            color = RED if f["severity"] in ("CRITICAL", "HIGH") else YELLOW
            print(f"{BOLD}[Finding {i}] [{color}{f['severity']}{RESET}{BOLD}] [{f['category']}]{RESET}")
            print(f"  Summary: {f['summary']}")
            print(f"  Details: {f['details']}\n")
        print("=" * 80)


REPORT = RedTeamAuditReport()


def log_probe(title: str):
    print(f"\n{BOLD}{CYAN}>>> [PROBE] {title}{RESET}")


# ============================================================================
# 1. MEMORY & ALIGNMENT INVARIANTS
# ============================================================================

def probe_strided_and_non_contiguous_views():
    """Attack Vector 1.1: Strided / Non-contiguous MLX views."""
    log_probe("1.1 Strided & Non-Contiguous MLX Views")
    REPORT.tests_run += 1

    X = mx.ones((64, 64), dtype=mx.float16)

    # 1. Non-contiguous slice: arr[:, ::2]
    strided_X = X[:, ::2]
    mx.eval(strided_X)

    try:
        # Extracting buffer from strided slice must auto-convert cleanly
        ptr, nbytes = MetalUMABridge.extract_buffer_address_and_size(strided_X)
        if ptr == 0 or nbytes != strided_X.nbytes:
            REPORT.record_finding(
                severity="HIGH",
                category="Memory Invariant",
                summary="extract_buffer_address_and_size returned invalid pointer or size for strided view",
                details=f"ptr={ptr}, nbytes={nbytes}, expected={strided_X.nbytes}"
            )
            return
        print(f"  [Pass] Non-contiguous strided slice auto-converted cleanly (ptr={ptr}, bytes={nbytes})")
    except Exception as e:
        REPORT.record_finding(
            severity="MEDIUM",
            category="Memory Invariant",
            summary="Strided MLX views failed to extract contiguous buffer",
            details=f"Exception raised: {type(e).__name__}: {e}"
        )
        return

    # 2. Contiguous sub-slice with non-zero byte offset
    sub_X = X[10:20, :] # 10 rows, contiguous, but offset > 0
    mx.eval(sub_X)
    ptr_sub, nbytes_sub = MetalUMABridge.extract_buffer_address_and_size(sub_X)
    ptr_base, _ = MetalUMABridge.extract_buffer_address_and_size(X)
    offset_bytes = ptr_sub - ptr_base
    expected_offset = 10 * 64 * 2
    assert offset_bytes == expected_offset, f"Offset mismatch: got {offset_bytes}, expected {expected_offset}"
    print(f"  [Pass] Contiguous row slice correctly identified offset: {offset_bytes} bytes")
    REPORT.tests_passed += 1


def probe_byte_alignment_and_metal_offset_math():
    """Attack Vector 1.2: Page alignment math & sub-element offsets."""
    log_probe("1.2 Sub-element Alignment & Metal Buffer Offset Constraints")
    REPORT.tests_run += 1

    bridge = MetalUMABridge.get_instance()
    buf = bytearray(65536)
    base_ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(buf))

    # An odd pointer offset (1 byte)
    odd_ptr = base_ptr + 1
    w_ptr = base_ptr + 1024
    y_ptr = base_ptr + 2048

    # The host bridge must reject odd-aligned pointers with status code -2
    code_x = bridge._lib.m4_bridge_dispatch_gemm(
        odd_ptr, 1024,
        w_ptr, 1024,
        y_ptr, 1024,
        0, 32, 32, 32
    )

    code_y = bridge._lib.m4_bridge_dispatch_gemm(
        base_ptr, 1024,
        w_ptr, 1024,
        odd_ptr, 1024,
        0, 32, 32, 32
    )

    if code_x == -2 and code_y == -2:
        print(f"  [Pass] Unaligned (odd) pointers for X and Y correctly rejected with status code -2.")
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="HIGH",
            category="Alignment Invariant",
            summary="Host C++ bridge does not enforce 2-byte alignment on pointers",
            details=f"Returned code_x={code_x}, code_y={code_y}, expected -2 for unaligned pointers."
        )


def probe_virtual_memory_page_bounds():
    """Attack Vector 1.3: Page alignment math and unmapped page boundaries."""
    log_probe("1.3 Page Alignment Math & Virtual Memory Boundary Overflow")
    REPORT.tests_run += 1

    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int

    page_size = 16384
    # Allocate 2 pages, protect page 2 as PROT_NONE
    buf = mmap.mmap(-1, page_size * 2, mmap.MAP_ANON | mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
    base_addr = ctypes.addressof((ctypes.c_char * 1).from_buffer(buf))
    libc.mprotect(base_addr + page_size, page_size, 0) # Guard page

    bridge = MetalUMABridge.get_instance()
    # If pointer starts inside Page 1 with length that spills into Guard Page
    x_ptr = base_addr + 100
    x_bytes = page_size # total_len = 100 + 16384 = 16484 > 16384
    w_bytes = 4096
    y_bytes = 4096
    w_buf = bytearray(w_bytes)
    y_buf = bytearray(y_bytes)
    w_ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(w_buf))
    y_ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(y_buf))

    ret = bridge._lib.m4_bridge_dispatch_gemm(
        x_ptr, x_bytes,
        w_ptr, w_bytes,
        y_ptr, y_bytes,
        0, 32, 64, 64
    )
    print(f"  [Observed] Spilling into unmapped/PROT_NONE page returned code: {ret}")
    if ret == -5:
        print("  [Pass] newBufferWithBytesNoCopy gracefully returned nil and bridge returned -5.")
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="HIGH",
            category="Memory Invariant",
            summary="VM boundary overflow did not return -5",
            details=f"Bridge returned status {ret} when wrapping memory that spans into an unmapped page."
        )


# ============================================================================
# 2. FORMAT COVERAGE & CODEC ROBUSTNESS
# ============================================================================

def probe_all_six_quant_formats():
    """Attack Vector 2.1: Test ALL 6 formats in QuantFormat."""
    log_probe("2.1 Coverage of All 6 Quantization Formats")
    REPORT.tests_run += 1

    formats = [
        (QuantFormat.QUANT_Q4_0, "QUANT_Q4_0", 32, 18, False),
        (QuantFormat.QUANT_MLX_4BIT, "QUANT_MLX_4BIT", 32, 20, False),
        (QuantFormat.QUANT_Q4_K, "QUANT_Q4_K", 256, 144, True),
        (QuantFormat.QUANT_TERNARY_1_58, "QUANT_TERNARY_1_58", 32, 12, False),
        (QuantFormat.QUANT_VAR_RATE_AFFINE, "QUANT_VAR_RATE_AFFINE", 256, 160, True),
        (QuantFormat.QUANT_EXL3, "QUANT_EXL3", 256, 144, True),
    ]

    M, K, N = 64, 256, 64
    X = mx.ones((M, K), dtype=mx.float16)

    all_ok = True
    for fmt_val, fmt_name, blk_weights, blk_bytes, is_super in formats:
        num_blocks = (K // blk_weights) * N
        w_bytes = num_blocks * blk_bytes
        W = mx.zeros((w_bytes,), dtype=mx.uint8)
        Y = mx.zeros((M, N), dtype=mx.float16)

        try:
            ret = m4_bridge_dispatch_gemm(X, W, Y, fmt_val, M, K, N)
            m4_bridge_synchronize()
            is_finite = bool(mx.all(mx.isfinite(Y)).item())
            print(f"  [Format {fmt_val}: {fmt_name:<24}] ret={ret}, Y shape={Y.shape}, finite={is_finite}")
            if ret != 0 or not is_finite:
                all_ok = False
        except Exception as e:
            print(f"  [Format {fmt_val}: {fmt_name:<24}] EXCEPTION: {e}")
            all_ok = False

    if all_ok:
        print("  [Pass] All 6 quantization formats executed successfully with finite outputs.")
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="HIGH",
            category="Codec Robustness",
            summary="One or more quantization formats failed execution",
            details="Failure encountered when dispatching GEMM across the 6 supported quantization formats."
        )


def probe_invalid_format_ids():
    """Attack Vector 2.2: Invalid format IDs."""
    log_probe("2.2 Invalid Format IDs (-1, 6, 999, -999)")
    REPORT.tests_run += 1

    M, K, N = 64, 64, 64
    X = mx.ones((M, K), dtype=mx.float16)
    W = mx.zeros((10000,), dtype=mx.uint8)
    Y = mx.zeros((M, N), dtype=mx.float16)

    invalid_ids = [-1, 6, 999, -999]
    all_caught = True
    for bad_id in invalid_ids:
        try:
            m4_bridge_dispatch_gemm(X, W, Y, bad_id, M, K, N)
            print(f"  [Fail] Format ID {bad_id} was NOT rejected!")
            all_caught = False
        except RuntimeError as e:
            print(f"  [Pass] Format ID {bad_id} rejected with: {e}")

    if all_caught:
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="HIGH",
            category="Codec Robustness",
            summary="Invalid format IDs were not rejected by bridge",
            details="m4_bridge_dispatch_gemm accepted format IDs outside the valid range [0, 5]."
        )


def probe_buffer_bounds_and_underrun():
    """Attack Vector 2.3: Buffer bounds validation vs Out-of-Bounds GPU reads/writes."""
    log_probe("2.3 Undersized Buffer Bounds & OOB GPU Access")
    REPORT.tests_run += 1

    M, K, N = 64, 64, 64
    bridge = MetalUMABridge.get_instance()

    # Create undersized buffers
    X_tiny = mx.ones((1, 1), dtype=mx.float16)  # 2 bytes, needs 8192 bytes
    W_tiny = mx.zeros((10,), dtype=mx.uint8)    # 10 bytes, needs 2304 bytes
    Y_tiny = mx.zeros((1, 1), dtype=mx.float16) # 2 bytes, needs 8192 bytes

    x_tiny_ptr, x_tiny_bytes = bridge.extract_buffer_address_and_size(X_tiny)
    w_tiny_ptr, w_tiny_bytes = bridge.extract_buffer_address_and_size(W_tiny)
    y_tiny_ptr, y_tiny_bytes = bridge.extract_buffer_address_and_size(Y_tiny)

    X_full = mx.ones((M, K), dtype=mx.float16)
    W_full = mx.zeros(((K // 32) * N * 18,), dtype=mx.uint8)
    Y_full = mx.zeros((M, N), dtype=mx.float16)

    x_full_ptr, x_full_bytes = bridge.extract_buffer_address_and_size(X_full)
    w_full_ptr, w_full_bytes = bridge.extract_buffer_address_and_size(W_full)
    y_full_ptr, y_full_bytes = bridge.extract_buffer_address_and_size(Y_full)

    # Test undersized X
    code_bad_x = bridge._lib.m4_bridge_dispatch_gemm(
        x_tiny_ptr, x_tiny_bytes, w_full_ptr, w_full_bytes, y_full_ptr, y_full_bytes, 0, M, K, N
    )
    # Test undersized W
    code_bad_w = bridge._lib.m4_bridge_dispatch_gemm(
        x_full_ptr, x_full_bytes, w_tiny_ptr, w_tiny_bytes, y_full_ptr, y_full_bytes, 0, M, K, N
    )
    # Test undersized Y
    code_bad_y = bridge._lib.m4_bridge_dispatch_gemm(
        x_full_ptr, x_full_bytes, w_full_ptr, w_full_bytes, y_tiny_ptr, y_tiny_bytes, 0, M, K, N
    )

    # Test via Python wrapper dispatch_gemm raising RuntimeError
    caught_wrapper = False
    try:
        bridge.dispatch_gemm(X_tiny, W_full, Y_full, QuantFormat.QUANT_Q4_0, M, K, N)
    except RuntimeError as e:
        if "-2" in str(e):
            caught_wrapper = True

    if code_bad_x == -2 and code_bad_w == -2 and code_bad_y == -2 and caught_wrapper:
        print("  [Pass] Undersized buffer bounds rejected systematically with status code -2.")
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="CRITICAL",
            category="Buffer Safety",
            summary="C++ bridge failed to reject undersized buffers with -2",
            details=f"code_bad_x={code_bad_x}, code_bad_w={code_bad_w}, code_bad_y={code_bad_y}, caught_wrapper={caught_wrapper}"
        )


def probe_k_alignment_for_super_blocks():
    """Attack Vector 2.4: Super-block formats with K not divisible by 256."""
    log_probe("2.4 Super-Block Formats (Q4_K, VAR_RATE, EXL3) with K % 256 != 0")
    REPORT.tests_run += 1

    M, K, N = 64, 128, 64 # K is divisible by 32, but NOT by 256
    X = mx.ones((M, K), dtype=mx.float16)
    W = mx.zeros((100000,), dtype=mx.uint8)
    Y = mx.zeros((M, N), dtype=mx.float16)

    bridge = MetalUMABridge.get_instance()
    all_rejected = True
    sb_formats = [
        (QuantFormat.QUANT_Q4_K, "QUANT_Q4_K"),
        (QuantFormat.QUANT_VAR_RATE_AFFINE, "QUANT_VAR_RATE_AFFINE"),
        (QuantFormat.QUANT_EXL3, "QUANT_EXL3"),
    ]

    for fmt_val, fmt_name in sb_formats:
        try:
            bridge.dispatch_gemm(X, W, Y, fmt_val, M, K, N)
            print(f"  [Fail] Format {fmt_name} with K={K} (K%256!=0) was NOT rejected!")
            all_rejected = False
        except RuntimeError as e:
            if "-4" in str(e):
                print(f"  [Pass] {fmt_name} with K={K} correctly rejected with status code -4.")
            else:
                print(f"  [Fail] {fmt_name} raised unexpected error: {e}")
                all_rejected = False

    if all_rejected:
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="CRITICAL",
            category="Codec Invariant",
            summary="Bridge permits K % 256 != 0 for 256-weight super-block formats",
            details="One or more super-block formats failed to reject K % 256 != 0 with code -4."
        )


# ============================================================================
# 3. EDGE CASE DIMENSION BOUNDARIES
# ============================================================================

def probe_dimension_boundaries():
    """Attack Vector 3.1 & 3.2: Dimension boundaries M, N, K."""
    log_probe("3. Dimension Boundaries (M in {1, 33, 127, 128, 129, 2048}, N/K not divisible by 64)")
    REPORT.tests_run += 1

    m_values = [1, 33, 127, 128, 129, 2048]
    K, N = 64, 64
    all_m_passed = True

    for M in m_values:
        X = mx.ones((M, K), dtype=mx.float16)
        num_blocks = (K // 32) * N
        w_bytes = num_blocks * 18
        w_raw = np.zeros(w_bytes, dtype=np.uint8)
        for b in range(num_blocks):
            w_raw[b * 18 + 0] = 0x00
            w_raw[b * 18 + 1] = 0x3C
            for q in range(16):
                w_raw[b * 18 + 2 + q] = 0x99
        W = mx.array(w_raw)
        Y = mx.zeros((M, N), dtype=mx.float16)

        ret = m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_Q4_0, M, K, N)
        m4_bridge_synchronize()

        sample = Y[0, 0].item()
        is_finite = bool(mx.all(mx.isfinite(Y)).item())
        match = abs(sample - float(K)) < 1e-2
        if ret != 0 or not is_finite or not match:
            all_m_passed = False
            print(f"  [Fail M={M}] ret={ret}, finite={is_finite}, sample={sample}, expected={float(K)}")
        else:
            print(f"  [Pass M={M:<4}] Output exact: {sample} == {float(K)}, finite={is_finite}")

    if all_m_passed:
        REPORT.tests_passed += 1

    # N not divisible by 64 (e.g. N=33, 65)
    for test_n in [33, 65]:
        M, test_k = 32, 64
        X = mx.ones((M, test_k), dtype=mx.float16)
        num_blocks = (test_k // 32) * test_n
        w_raw = np.zeros(num_blocks * 18, dtype=np.uint8)
        for b in range(num_blocks):
            w_raw[b * 18 + 0] = 0x00
            w_raw[b * 18 + 1] = 0x3C
            for q in range(16):
                w_raw[b * 18 + 2 + q] = 0x99
        W = mx.array(w_raw)
        Y = mx.zeros((M, test_n), dtype=mx.float16)

        m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_Q4_0, M, test_k, test_n)
        m4_bridge_synchronize()
        sample = Y[0, test_n - 1].item()
        assert abs(sample - float(test_k)) < 1e-2, f"Expected {test_k}, got {sample}"
        print(f"  [Pass N={test_n:<4}] Boundary column {test_n-1} output exact: {sample}")

    # K non-divisible by 32 rejection
    rejected_bad_k = False
    try:
        m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_Q4_0, 32, 48, 64)
    except RuntimeError:
        rejected_bad_k = True
        print("  [Pass] K=48 (not divisible by 32) correctly rejected with status code -4.")

    if not rejected_bad_k:
        REPORT.record_finding(
            severity="HIGH",
            category="Dimension Invariant",
            summary="K not divisible by 32 was not rejected",
            details="m4_bridge_dispatch_gemm must reject K % 32 != 0."
        )


# ============================================================================
# 4. GARBAGE COLLECTION & RACE CONDITIONS
# ============================================================================

def probe_gc_lifecycle_race():
    """Attack Vector 4: Python GC freeing buffers while GPU kernel is in-flight."""
    log_probe("4. Garbage Collection & In-Flight Buffer Lifespan")
    REPORT.tests_run += 1

    M, K, N = 512, 512, 512
    bridge = MetalUMABridge.get_instance()

    X_e = mx.ones((M, K), dtype=mx.float16)
    nb = (K // 32) * N
    W_e = mx.zeros((nb * 18,), dtype=mx.uint8)
    Y_e = mx.zeros((M, N), dtype=mx.float16)

    m4_bridge_dispatch_gemm(X_e, W_e, Y_e, QuantFormat.QUANT_Q4_0, M, K, N)

    # Verify that in-flight tensors are retained by the bridge
    retained_count_before = len(bridge._in_flight_tensors)
    assert retained_count_before > 0, "Bridge must retain in-flight tensors before synchronize()"

    # Delete local caller references and force GC collection
    del X_e
    del W_e
    del Y_e
    gc.collect()

    # Synchronize and verify release
    m4_bridge_synchronize()
    retained_count_after = len(bridge._in_flight_tensors)

    if retained_count_before > 0 and retained_count_after == 0:
        print(f"  [Pass] In-flight tensors safely retained during GPU compute ({retained_count_before} entries) and cleared upon synchronize ({retained_count_after} entries).")
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="HIGH",
            category="Concurrency & Lifetime",
            summary="Python bridge does not retain references to in-flight buffers until synchronize()",
            details=f"retained_before={retained_count_before}, retained_after={retained_count_after}"
        )


# ============================================================================
# 5. METROLOGY COMPLIANCE & TRIPWIRES
# ============================================================================

def probe_metrology_and_tripwires():
    """Attack Vector 5: Metrology UMA footprint and NaN/Inf tripwires."""
    log_probe("5. Metrology Compliance (task_vm_info.phys_footprint & NaN Tripwires)")
    REPORT.tests_run += 1

    footprint = m4_bridge_get_uma_footprint_mb()
    print(f"  [UMA Telemetry] task_vm_info.phys_footprint: {footprint:.2f} MB")
    assert footprint > 0.0, "UMA footprint must be positive"

    # NaN / Inf Tripwires:
    M, K, N = 64, 64, 64
    X = mx.ones((M, K), dtype=mx.float16)
    X[0, 0] = float('nan') # Inject NaN into activation tensor
    W = mx.zeros(((K//32)*N*18,), dtype=mx.uint8)
    Y = mx.zeros((M, N), dtype=mx.float16)

    # 1. Test check_finite=True with injected NaN -> must raise FloatingPointError
    m4_bridge_dispatch_gemm(X, W, Y, QuantFormat.QUANT_Q4_0, M, K, N, check_finite=True)
    tripwire_triggered = False
    try:
        m4_bridge_synchronize()
    except FloatingPointError as e:
        tripwire_triggered = True
        print(f"  [Pass] NaN tripwire triggered with check_finite=True: {e}")

    # 2. Test check_finite=True with valid finite tensor -> must pass cleanly
    X_clean = mx.ones((M, K), dtype=mx.float16)
    Y_clean = mx.zeros((M, N), dtype=mx.float16)
    m4_bridge_dispatch_gemm(X_clean, W, Y_clean, QuantFormat.QUANT_Q4_0, M, K, N, check_finite=True)
    clean_passed = False
    try:
        m4_bridge_synchronize()
        clean_passed = True
        print("  [Pass] Finite output verified without false positives under check_finite=True.")
    except Exception as e:
        print(f"  [Fail] Unexpected exception on finite output: {e}")

    if tripwire_triggered and clean_passed:
        REPORT.tests_passed += 1
    else:
        REPORT.record_finding(
            severity="MEDIUM",
            category="Metrology Tripwire",
            summary="Bridge lacks working NaN/Inf tripwire assertion",
            details=f"tripwire_triggered={tripwire_triggered}, clean_passed={clean_passed}"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print(f"{BOLD}STARTING RIGOROUS STEP 1 RED-TEAM AUDIT OF METAL UMA BRIDGE{RESET}")
    ok = m4_bridge_init()
    assert ok, "Failed to initialize Metal bridge"

    probe_strided_and_non_contiguous_views()
    probe_byte_alignment_and_metal_offset_math()
    probe_virtual_memory_page_bounds()
    probe_all_six_quant_formats()
    probe_invalid_format_ids()
    probe_buffer_bounds_and_underrun()
    probe_k_alignment_for_super_blocks()
    probe_dimension_boundaries()
    probe_gc_lifecycle_race()
    probe_metrology_and_tripwires()

    REPORT.print_summary()


if __name__ == "__main__":
    main()
