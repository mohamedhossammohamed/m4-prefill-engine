"""
Basic Verification Test Suite for Pluggable M4 vs. MLX Module Pair.

Validates:
1. M4QuantizedLinear forward pass under prefill (M > 1) and decode (M = 1)
   across multiple quantization formats (Q4_0, MLX_4BIT, TERNARY_1_58, Q4_K).
2. Numerical correctness, shape propagation, and finiteness (no NaN/Inf).
3. Parity comparison between M4 accelerated kernels and MLX dequantized baseline.
4. M4KVCache incremental token insertion, retrieval, in-RAM circular pre-allocation,
   and NVMe out-of-core flash streaming.
5. MLXKVCache standard baseline operation.
6. Factory functions (create_linear, create_kv_cache) for modular backend switching.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import unittest

from core.bridge.m4_bridge import m4_bridge_synchronize
from src.engine import (
    FORMAT_SPECS,
    M4KVCache,
    M4QuantizedLinear,
    MLXKVCache,
    MLXQuantizedLinear,
    QuantFormat,
    create_kv_cache,
    create_linear,
    generate_synthetic_quantized_weights,
)


def test_m4_quantized_linear_prefill_and_decode_formats():
    """Verify M4QuantizedLinear across Q4_0, MLX_4BIT, and TERNARY_1_58."""
    formats_to_test = [
        QuantFormat.QUANT_Q4_0,
        QuantFormat.QUANT_MLX_4BIT,
        QuantFormat.QUANT_TERNARY_1_58,
    ]
    K, N = 64, 64

    for fmt in formats_to_test:
        layer = M4QuantizedLinear(in_features=K, out_features=N, format=fmt)
        layer.init_synthetic_weights(scale=1.0)

        # 1. Prefill Forward Pass (M > 1)
        for M in [2, 33, 64, 128]:
            x_prefill = mx.ones((M, K), dtype=mx.float16)
            y_prefill = layer(x_prefill)
            m4_bridge_synchronize()

            assert y_prefill.shape == (M, N), f"Expected shape {(M, N)}, got {y_prefill.shape}"
            assert y_prefill.dtype == mx.float16, f"Expected float16, got {y_prefill.dtype}"
            assert bool(mx.all(mx.isfinite(y_prefill)).item()) is True, f"Non-finite in prefill {fmt.name}"

            # Check deterministic expected dot product
            sample = y_prefill[0, 0].item()
            assert abs(sample - float(K)) < 1e-1, f"Expected ~{float(K)}, got {sample} in {fmt.name}"

        # 2. Decode Forward Pass (M = 1)
        x_decode = mx.ones((1, K), dtype=mx.float16)
        y_decode = layer(x_decode)
        m4_bridge_synchronize()

        assert y_decode.shape == (1, N), f"Expected shape {(1, N)}, got {y_decode.shape}"
        assert y_decode.dtype == mx.float16
        assert bool(mx.all(mx.isfinite(y_decode)).item()) is True, f"Non-finite in decode {fmt.name}"

        sample_dec = y_decode[0, 0].item()
        assert abs(sample_dec - float(K)) < 1e-1, f"Expected ~{float(K)}, got {sample_dec} in {fmt.name}"

        # 3. 1D Decode Forward Pass [K]
        x_1d = mx.ones((K,), dtype=mx.float16)
        y_1d = layer(x_1d)
        m4_bridge_synchronize()
        assert y_1d.shape == (N,)
        assert bool(mx.all(mx.isfinite(y_1d)).item()) is True

        print(f"[PASS] M4QuantizedLinear verified for {fmt.name} (prefill + decode)")


def test_m4_quantized_linear_super_block_format():
    """Verify Q4_K super-block format (256-element block)."""
    K, N = 256, 64
    layer = M4QuantizedLinear(in_features=K, out_features=N, format=QuantFormat.QUANT_Q4_K)

    # Prefill M > 1
    x_prefill = mx.ones((16, K), dtype=mx.float16)
    y_prefill = layer(x_prefill)
    m4_bridge_synchronize()
    assert y_prefill.shape == (16, N)
    assert bool(mx.all(mx.isfinite(y_prefill)).item()) is True

    # Decode M = 1
    x_decode = mx.ones((1, K), dtype=mx.float16)
    y_decode = layer(x_decode)
    m4_bridge_synchronize()
    assert y_decode.shape == (1, N)
    assert bool(mx.all(mx.isfinite(y_decode)).item()) is True
    print("[PASS] M4QuantizedLinear verified for QUANT_Q4_K")


def test_m4_quantized_linear_3d_batching():
    """Verify 3D [B, M, K] tensor batching for both prefill and decode."""
    K, N = 64, 64
    layer = M4QuantizedLinear(in_features=K, out_features=N, format=QuantFormat.QUANT_Q4_0)
    layer.init_synthetic_weights(scale=1.0)

    # 1. Single-batch decode [1, 1, K]
    x_b1_m1 = mx.ones((1, 1, K), dtype=mx.float16)
    y_b1_m1 = layer(x_b1_m1)
    m4_bridge_synchronize()
    assert y_b1_m1.shape == (1, 1, N)
    assert bool(mx.all(mx.isfinite(y_b1_m1)).item()) is True

    # 2. Multi-batch decode [4, 1, K]
    x_b4_m1 = mx.ones((4, 1, K), dtype=mx.float16)
    y_b4_m1 = layer(x_b4_m1)
    m4_bridge_synchronize()
    assert y_b4_m1.shape == (4, 1, N)
    assert bool(mx.all(mx.isfinite(y_b4_m1)).item()) is True

    # 3. Batched prefill [2, 16, K]
    x_b2_m16 = mx.ones((2, 16, K), dtype=mx.float16)
    y_b2_m16 = layer(x_b2_m16)
    m4_bridge_synchronize()
    assert y_b2_m16.shape == (2, 16, N)
    assert bool(mx.all(mx.isfinite(y_b2_m16)).item()) is True
    print("[PASS] M4QuantizedLinear 3D tensor batching verified")


def test_m4_vs_mlx_numerical_parity():
    """Verify numerical parity between M4 hardware acceleration and MLX dequantized baseline."""
    K, N = 64, 64
    M = 4

    for fmt in [QuantFormat.QUANT_Q4_0, QuantFormat.QUANT_MLX_4BIT, QuantFormat.QUANT_TERNARY_1_58]:
        w_raw = generate_synthetic_quantized_weights(fmt, K, N, scale=1.0, seed=123)

        m4_layer = M4QuantizedLinear.from_quantized(w_raw, format=fmt, in_features=K, out_features=N)
        mlx_layer = MLXQuantizedLinear.from_quantized(w_raw, format=fmt, in_features=K, out_features=N)

        x = mx.random.normal((M, K)).astype(mx.float16)

        y_m4 = m4_layer(x)
        y_mlx = mlx_layer(x)
        m4_bridge_synchronize()

        assert y_m4.shape == y_mlx.shape
        assert bool(mx.all(mx.isfinite(y_m4)).item()) is True
        assert bool(mx.all(mx.isfinite(y_mlx)).item()) is True

        max_err = mx.max(mx.abs(y_m4 - y_mlx)).item()
        assert max_err < 0.1, f"Parity error too high for {fmt.name}: {max_err}"
        print(f"[PASS] M4 vs. MLX parity verified for {fmt.name} (max_diff={max_err:.6f})")


def test_m4_kv_cache_in_ram_incremental_and_retrieval():
    """Verify M4KVCache incremental token addition, retrieval, and zero allocation churn."""
    H, D = 4, 64
    cache = M4KVCache(head_dim=D, n_heads=H, max_seq_len=64, mode="in_ram")

    # 1. Prefill Step (Prompt of 16 tokens)
    prompt_len = 16
    k_prompt = mx.random.normal((1, H, prompt_len, D)).astype(mx.float16)
    v_prompt = mx.random.normal((1, H, prompt_len, D)).astype(mx.float16)

    k_out, v_out = cache.update_and_fetch(k_prompt, v_prompt)
    assert k_out.shape == (1, H, prompt_len, D)
    assert v_out.shape == (1, H, prompt_len, D)
    assert cache.offset == prompt_len
    assert bool(mx.all(mx.isfinite(k_out)).item()) is True
    assert bool(mx.all(k_out == k_prompt).item()) is True

    # 2. Autoregressive Decode Steps (One token at a time)
    decode_steps = 10
    inserted_keys = [k_prompt]
    inserted_vals = [v_prompt]

    for step in range(decode_steps):
        k_tok = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v_tok = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        inserted_keys.append(k_tok)
        inserted_vals.append(v_tok)

        k_ctx, v_ctx = cache.update_and_fetch(k_tok, v_tok)
        expected_len = prompt_len + step + 1

        assert k_ctx.shape == (1, H, expected_len, D)
        assert v_ctx.shape == (1, H, expected_len, D)
        assert cache.offset == expected_len
        assert bool(mx.all(mx.isfinite(k_ctx)).item()) is True

        # Verify most recently inserted token matches slice
        assert bool(mx.all(k_ctx[:, :, -1:, :] == k_tok).item()) is True
        assert bool(mx.all(v_ctx[:, :, -1:, :] == v_tok).item()) is True

    # Verify complete context matches full concatenation
    full_k_ref = mx.concatenate(inserted_keys, axis=2)
    assert bool(mx.all(k_ctx == full_k_ref).item()) is True

    # 3. Test Reset
    cache.reset()
    assert cache.offset == 0
    k_new = mx.ones((1, H, 4, D), dtype=mx.float16)
    v_new = mx.ones((1, H, 4, D), dtype=mx.float16)
    k_res, v_res = cache.update_and_fetch(k_new, v_new)
    assert k_res.shape == (1, H, 4, D)
    assert cache.offset == 4
    print("[PASS] M4KVCache in_ram incremental insertion and exact retrieval verified")


def test_m4_kv_cache_dynamic_expansion():
    """Verify M4KVCache dynamically expands buffer if sequence exceeds initial max_seq_len."""
    H, D = 2, 32
    cache = M4KVCache(head_dim=D, n_heads=H, max_seq_len=8, mode="in_ram")

    # Insert 16 tokens into capacity-8 cache
    k = mx.ones((1, H, 16, D), dtype=mx.float16)
    v = mx.ones((1, H, 16, D), dtype=mx.float16)
    k_out, v_out = cache.update_and_fetch(k, v)

    assert k_out.shape == (1, H, 16, D)
    assert cache.offset == 16
    assert bool(mx.all(mx.isfinite(k_out)).item()) is True
    print("[PASS] M4KVCache dynamic buffer expansion verified")


def test_m4_kv_cache_out_of_core_mode():
    """Verify M4KVCache out_of_core mode streaming to NVMe with zero disk litter."""
    H, D = 4, 32
    cache_file_path = None
    with M4KVCache(head_dim=D, n_heads=H, mode="out_of_core", ram_capacity=10) as cache:
        cache_file_path = cache._file_path
        assert cache_file_path is not None
        assert cache_file_path.is_file()

        # Insert 15 tokens: active RAM window should retain latest 10 tokens
        k = mx.ones((1, H, 15, D), dtype=mx.float16)
        v = mx.ones((1, H, 15, D), dtype=mx.float16)
        k_out, v_out = cache.update_and_fetch(k, v)

        assert cache.offset == 15
        assert k_out.shape == (1, H, 10, D)
        assert bool(mx.all(mx.isfinite(k_out)).item()) is True

    # Verify zero disk litter cleanup upon context manager exit
    assert not cache_file_path.exists()
    print("[PASS] M4KVCache out_of_core NVMe streaming & zero disk litter verified")


def test_mlx_kv_cache_baseline():
    """Verify MLXKVCache standard baseline operation."""
    H, D = 4, 64
    cache = MLXKVCache(head_dim=D, n_heads=H)

    k1 = mx.ones((1, H, 12, D), dtype=mx.float16)
    v1 = mx.ones((1, H, 12, D), dtype=mx.float16)
    k_out, v_out = cache.update_and_fetch(k1, v1)
    assert k_out.shape == (1, H, 12, D)
    assert cache.offset == 12

    k2 = mx.full((1, H, 1, D), 2.0, dtype=mx.float16)
    v2 = mx.full((1, H, 1, D), 2.0, dtype=mx.float16)
    k_out, v_out = cache.update_and_fetch(k2, v2)
    assert k_out.shape == (1, H, 13, D)
    assert cache.offset == 13
    assert bool(mx.all(mx.isfinite(k_out)).item()) is True

    cache.reset()
    assert cache.offset == 0
    print("[PASS] MLXKVCache baseline verified")


def test_factory_functions():
    """Verify create_linear and create_kv_cache factory functions."""
    # 1. create_linear defaults to m4 backend
    linear_default = create_linear(64, 64)
    assert isinstance(linear_default, M4QuantizedLinear)

    linear_m4 = create_linear(64, 64, backend="m4")
    assert isinstance(linear_m4, M4QuantizedLinear)

    linear_mlx = create_linear(64, 64, backend="mlx")
    assert isinstance(linear_mlx, MLXQuantizedLinear)

    try:
        create_linear(64, 64, backend="cuda")
        assert False, "Should have raised ValueError for invalid backend"
    except ValueError:
        pass

    # 2. create_kv_cache defaults to m4 backend
    cache_default = create_kv_cache(head_dim=64, n_heads=4)
    assert isinstance(cache_default, M4KVCache)
    assert cache_default.mode == "in_ram"

    cache_m4_ooc = create_kv_cache(head_dim=64, n_heads=4, backend="m4", mode="out_of_core")
    assert isinstance(cache_m4_ooc, M4KVCache)
    assert cache_m4_ooc.mode == "out_of_core"
    cache_m4_ooc.close()

    cache_mlx = create_kv_cache(head_dim=64, n_heads=4, backend="mlx")
    assert isinstance(cache_mlx, MLXKVCache)

    try:
        create_kv_cache(head_dim=64, n_heads=4, backend="tpu")
        assert False, "Should have raised ValueError for invalid backend"
    except ValueError:
        pass

    print("[PASS] Factory functions verified for both backends")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TEST SUITE: PLUGGABLE M4 VS. MLX MODULE PAIR")
    print("=" * 80)

    test_m4_quantized_linear_prefill_and_decode_formats()
    test_m4_quantized_linear_super_block_format()
    test_m4_quantized_linear_3d_batching()
    test_m4_vs_mlx_numerical_parity()
    test_m4_kv_cache_in_ram_incremental_and_retrieval()
    test_m4_kv_cache_dynamic_expansion()
    test_m4_kv_cache_out_of_core_mode()
    test_mlx_kv_cache_baseline()
    test_factory_functions()

    print("\n" + "=" * 80)
    print("[✓] ALL PLUGGABLE MODULE TESTS PASSED CLEANLY (EXIT CODE 0)!")
    print("=" * 80)
