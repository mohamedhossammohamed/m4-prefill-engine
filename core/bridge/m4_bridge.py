"""
Zero-Copy Metal UMA MLX Bridge Python Interface.

Directly bridges MLX arrays in unified memory with low-level Metal shaders
without host-device memory copying using ctypes and Python's buffer protocol.
"""

from __future__ import annotations
import ctypes
import os
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx


class QuantFormat(IntEnum):
    QUANT_Q4_0 = 0            # Standard 32-element symmetric block (18 bytes / block)
    QUANT_MLX_4BIT = 1        # MLX-style 32-element affine block (20 bytes / block)
    QUANT_Q4_K = 2            # GGUF-style 256-element super-block (144 bytes / super-block)
    QUANT_TERNARY_1_58 = 3    # BitNet-style 32-element ternary block (12 bytes / block)
    QUANT_VAR_RATE_AFFINE = 4 # Grouped Variable-Rate Affine 256-element super-block (160 bytes)
    QUANT_EXL3 = 5            # ExLlamaV3 256-element vector codebook super-block (144 bytes)


def _find_dylib_path() -> Path:
    env_path = os.environ.get("M4_BRIDGE_DYLIB_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    candidates = [
        Path(__file__).resolve().parent / "libm4_bridge.dylib",
        Path(__file__).resolve().parent.parent.parent / "libm4_bridge.dylib",
        Path.cwd() / "libm4_bridge.dylib",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not locate libm4_bridge.dylib. Checked: {[str(c) for c in candidates]}. "
        "Please compile via 'make libm4_bridge.dylib' or set M4_BRIDGE_DYLIB_PATH."
    )


class MetalUMABridge:
    _instance: Optional["MetalUMABridge"] = None
    _retained_contig: list[mx.array] = []

    def __init__(self, dylib_path: Optional[Union[str, Path]] = None) -> None:
        lib_path = Path(dylib_path) if dylib_path else _find_dylib_path()
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_bindings()
        self._in_flight_tensors: list[tuple[mx.array, ...]] = []
        self._pending_finite_checks: list[mx.array] = []

    def _setup_bindings(self) -> None:
        # bool m4_bridge_init(const char* metallib_path)
        self._lib.m4_bridge_init.argtypes = [ctypes.c_char_p]
        self._lib.m4_bridge_init.restype = ctypes.c_bool

        # int m4_bridge_dispatch_gemm(void* X_ptr, size_t X_bytes, void* W_ptr, size_t W_bytes,
        #                             void* Y_ptr, size_t Y_bytes, int format,
        #                             uint32_t M, uint32_t K, uint32_t N)
        self._lib.m4_bridge_dispatch_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        self._lib.m4_bridge_dispatch_gemm.restype = ctypes.c_int

        # int m4_bridge_dispatch_gemv(void* x_ptr, size_t x_bytes, void* W_ptr, size_t W_bytes,
        #                             void* y_ptr, size_t y_bytes, int format,
        #                             uint32_t K, uint32_t N)
        self._lib.m4_bridge_dispatch_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_uint32, ctypes.c_uint32,
        ]
        self._lib.m4_bridge_dispatch_gemv.restype = ctypes.c_int

        # void m4_bridge_synchronize()
        self._lib.m4_bridge_synchronize.argtypes = []
        self._lib.m4_bridge_synchronize.restype = None

        # double m4_bridge_get_uma_footprint_mb()
        self._lib.m4_bridge_get_uma_footprint_mb.argtypes = []
        self._lib.m4_bridge_get_uma_footprint_mb.restype = ctypes.c_double

        # bool m4_bridge_is_initialized()
        self._lib.m4_bridge_is_initialized.argtypes = []
        self._lib.m4_bridge_is_initialized.restype = ctypes.c_bool

        # void m4_bridge_shutdown()
        self._lib.m4_bridge_shutdown.argtypes = []
        self._lib.m4_bridge_shutdown.restype = None

    @classmethod
    def get_instance(cls, dylib_path: Optional[Union[str, Path]] = None) -> "MetalUMABridge":
        if cls._instance is None:
            cls._instance = cls(dylib_path)
        return cls._instance

    @classmethod
    def extract_buffer_address_and_size(cls, arr: mx.array) -> tuple[int, int]:
        """
        Extracts the unified memory pointer and byte size from an MLX array zero-copy.
        Ensures MLX evaluation and synchronization are complete so GPU memory is valid.
        If non-contiguous, auto-converts via mx.contiguous(arr) so it is safely C-contiguous.
        """
        mx.eval(arr)
        mx.synchronize()
        try:
            ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(arr))
            return ptr, arr.nbytes
        except (TypeError, BufferError):
            contig_arr = mx.contiguous(arr)
            mx.eval(contig_arr)
            mx.synchronize()
            cls._retained_contig.append(contig_arr)
            ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(contig_arr))
            return ptr, contig_arr.nbytes

    def init(self, metallib_path: Optional[str] = None) -> bool:
        c_path = metallib_path.encode("utf-8") if metallib_path else None
        return bool(self._lib.m4_bridge_init(c_path))

    def is_initialized(self) -> bool:
        return bool(self._lib.m4_bridge_is_initialized())

    def dispatch_gemm(
        self,
        X: mx.array,
        W: mx.array,
        Y: mx.array,
        format: Union[int, QuantFormat],
        M: Optional[int] = None,
        K: Optional[int] = None,
        N: Optional[int] = None,
        check_finite: bool = False,
    ) -> int:
        """
        Dispatches prefill GEMM on Metal shaders using zero-copy wrapped MLX unified buffers.
        """
        x_ptr, x_bytes = self.extract_buffer_address_and_size(X)
        w_ptr, w_bytes = self.extract_buffer_address_and_size(W)
        y_ptr, y_bytes = self.extract_buffer_address_and_size(Y)

        fmt_int = int(format)

        # Infer dimensions if omitted
        if M is None:
            M = X.shape[0] if len(X.shape) >= 2 else 1
        if K is None:
            K = X.shape[1] if len(X.shape) >= 2 else X.shape[0]
        if N is None:
            N = Y.shape[1] if len(Y.shape) >= 2 else Y.shape[0]

        ret = self._lib.m4_bridge_dispatch_gemm(
            x_ptr, x_bytes,
            w_ptr, w_bytes,
            y_ptr, y_bytes,
            fmt_int,
            int(M), int(K), int(N),
        )
        if ret != 0:
            raise RuntimeError(f"m4_bridge_dispatch_gemm failed with status code {ret}")

        # Retain references to in-flight tensors until synchronize()
        self._in_flight_tensors.append((X, W, Y))
        if check_finite:
            self._pending_finite_checks.append(Y)

        return ret

    def dispatch_gemv(
        self,
        x: mx.array,
        W: mx.array,
        y: mx.array,
        format: Union[int, QuantFormat],
        K: Optional[int] = None,
        N: Optional[int] = None,
        check_finite: bool = False,
    ) -> int:
        """
        Dispatches decode GEMV on Metal shaders using zero-copy wrapped MLX unified buffers.
        """
        x_ptr, x_bytes = self.extract_buffer_address_and_size(x)
        w_ptr, w_bytes = self.extract_buffer_address_and_size(W)
        y_ptr, y_bytes = self.extract_buffer_address_and_size(y)

        fmt_int = int(format)

        if K is None:
            K = x.shape[-1]
        if N is None:
            N = y.shape[-1]

        ret = self._lib.m4_bridge_dispatch_gemv(
            x_ptr, x_bytes,
            w_ptr, w_bytes,
            y_ptr, y_bytes,
            fmt_int,
            int(K), int(N),
        )
        if ret != 0:
            raise RuntimeError(f"m4_bridge_dispatch_gemv failed with status code {ret}")

        # Retain references to in-flight tensors until synchronize()
        self._in_flight_tensors.append((x, W, y))
        if check_finite:
            self._pending_finite_checks.append(y)

        return ret

    def synchronize(self) -> None:
        self._lib.m4_bridge_synchronize()
        self._in_flight_tensors.clear()
        self._retained_contig.clear()

        if self._pending_finite_checks:
            checks = list(self._pending_finite_checks)
            self._pending_finite_checks.clear()
            for tensor in checks:
                mx.eval(tensor)
                if not bool(mx.all(mx.isfinite(tensor)).item()):
                    raise FloatingPointError("Tripwire assertion failed: Non-finite values (NaN/Inf) detected in output tensor.")

    def get_uma_footprint_mb(self) -> float:
        return float(self._lib.m4_bridge_get_uma_footprint_mb())

    def shutdown(self) -> None:
        self._in_flight_tensors.clear()
        self._retained_contig.clear()
        self._pending_finite_checks.clear()
        self._lib.m4_bridge_shutdown()


# Module-level convenience functions matching C-ABI exports
def m4_bridge_init(metallib_path: Optional[str] = None) -> bool:
    return MetalUMABridge.get_instance().init(metallib_path)


def m4_bridge_dispatch_gemm(
    X: mx.array,
    W: mx.array,
    Y: mx.array,
    format: Union[int, QuantFormat],
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    check_finite: bool = False,
) -> int:
    return MetalUMABridge.get_instance().dispatch_gemm(X, W, Y, format, M, K, N, check_finite=check_finite)


def m4_bridge_dispatch_gemv(
    x: mx.array,
    W: mx.array,
    y: mx.array,
    format: Union[int, QuantFormat],
    K: Optional[int] = None,
    N: Optional[int] = None,
    check_finite: bool = False,
) -> int:
    return MetalUMABridge.get_instance().dispatch_gemv(x, W, y, format, K, N, check_finite=check_finite)


def m4_bridge_synchronize() -> None:
    MetalUMABridge.get_instance().synchronize()


def m4_bridge_get_uma_footprint_mb() -> float:
    return MetalUMABridge.get_instance().get_uma_footprint_mb()

