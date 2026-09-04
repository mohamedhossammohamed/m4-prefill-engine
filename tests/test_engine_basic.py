"""
Basic Verification Test Suite for Unified Engine and Inference Orchestrator.

Validates:
1. InferenceEngine initialization with backend="m4" (default) and backend="mlx".
2. Multi-token prompt prefill (M=16) and KV cache updates.
3. Autoregressive generation for 10 steps.
4. Sampling modes (greedy argmax vs. temperature vs. top-p).
5. Reset and subsequent generation reproducibility.
6. Streaming callback invocations during generation.
7. Memory footprint reporting via MetalUMABridge.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np
import unittest

from src.engine import (
    EngineConfig,
    InferenceEngine,
    QuantFormat,
    TransformerBlock,
    TransformerModel,
)


def test_engine_initialization():
    """Verify InferenceEngine initialization with backend='m4' and backend='mlx'."""
    # 1. Default initialization (backend="m4")
    engine_m4 = InferenceEngine()
    assert engine_m4.config.backend == "m4", f"Expected backend 'm4', got '{engine_m4.config.backend}'"
    assert engine_m4.config.num_layers == 4
    assert engine_m4.config.hidden_dim == 512
    assert engine_m4.config.num_q_heads == 8
    assert engine_m4.config.num_kv_heads == 2
    assert engine_m4.config.head_dim == 64
    assert engine_m4.config.intermediate_dim == 1024
    assert engine_m4.config.vocab_size == 1024
    assert engine_m4.config.kv_mode == "in_ram"
    assert isinstance(engine_m4.model, TransformerModel)
    assert len(engine_m4.model.layers) == 4

    # Memory footprint via MetalUMABridge
    footprint = engine_m4.get_memory_footprint_mb()
    assert footprint > 0.0, f"Expected positive UMA footprint, got {footprint}"
    print(f"[PASS] InferenceEngine initialized with backend='m4' (UMA footprint: {footprint:.2f} MB)")

    # 2. MLX backend initialization
    cfg_mlx = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="mlx",
    )
    engine_mlx = InferenceEngine(cfg_mlx)
    assert engine_mlx.config.backend == "mlx"
    assert len(engine_mlx.model.layers) == 2
    print("[PASS] InferenceEngine initialized with backend='mlx'")


def test_multi_token_prompt_prefill():
    """Verify multi-token prompt prefill (M=16) and proper KV caching."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    M = 16
    prompt_tokens = [int(i % cfg.vocab_size) for i in range(1, M + 1)]

    # 1. Prefill with list[int]
    logits = engine.prefill(prompt_tokens)

    assert isinstance(logits, mx.array), f"Expected mx.array, got {type(logits)}"
    assert logits.shape == (cfg.vocab_size,), f"Expected shape ({cfg.vocab_size},), got {logits.shape}"
    assert bool(mx.all(mx.isfinite(logits)).item()) is True, "Non-finite values found in prefill logits"
    assert engine.sequence_length == M, f"Expected sequence_length={M}, got {engine.sequence_length}"

    # Verify KV caches in all layers stored M tokens
    for idx, layer in enumerate(engine.model.layers):
        assert layer.kv_cache.offset == M, f"Layer {idx} KV cache offset mismatch: {layer.kv_cache.offset} != {M}"

    print(f"[PASS] Multi-token prompt prefill verified (M={M}, shape={logits.shape})")

    # 2. Prefill with mx.array
    engine.reset()
    tokens_mx = mx.array(prompt_tokens, dtype=mx.int32)
    logits_mx = engine.prefill(tokens_mx)
    assert logits_mx.shape == (cfg.vocab_size,)
    assert bool(mx.all(mx.isfinite(logits_mx)).item()) is True
    assert engine.sequence_length == M
    print("[PASS] Prompt prefill verified with direct mx.array input")


def test_autoregressive_generation_10_steps():
    """Verify autoregressive generation for exactly 10 steps."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    prompt = [12, 34, 56, 78]
    prompt_len = len(prompt)
    max_new_tokens = 10

    generated_tokens = engine.generate(
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    assert len(generated_tokens) == max_new_tokens, (
        f"Expected {max_new_tokens} tokens, got {len(generated_tokens)}"
    )
    assert all(isinstance(t, int) for t in generated_tokens), "All tokens must be python integers"
    assert all(0 <= t < cfg.vocab_size for t in generated_tokens), "Tokens out of vocabulary range"

    # Total sequence length in KV cache after generate():
    # Prompt (prompt_len) + (max_new_tokens - 1) processed decode tokens.
    # The last sampled token has not yet been fed into decode_step.
    expected_cache_len = prompt_len + max_new_tokens - 1
    assert engine.sequence_length == expected_cache_len, (
        f"Expected cache sequence length {expected_cache_len}, got {engine.sequence_length}"
    )

    # Check that individual decode_step also functions incrementally and appends last token
    next_step_token = engine.decode_step(generated_tokens[-1], temperature=0.0)
    assert isinstance(next_step_token, int)
    assert 0 <= next_step_token < cfg.vocab_size
    assert engine.sequence_length == prompt_len + max_new_tokens

    print(f"[PASS] Autoregressive generation verified (10 steps: {generated_tokens})")


def test_sampling_modes():
    """Verify greedy argmax, temperature scaling, and top-p sampling."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    # Synthetic logits with clear dominant token at index 42
    logits = np.zeros(512, dtype=np.float32)
    logits[42] = 10.0
    logits[100] = 5.0
    logits[200] = 2.0
    logits_mx = mx.array(logits)

    # 1. Greedy Argmax (temperature=0.0)
    for _ in range(5):
        greedy_token = engine.sample(logits_mx, temperature=0.0)
        assert greedy_token == 42, f"Greedy sample failed: expected 42, got {greedy_token}"

    print("[PASS] Greedy argmax sampling verified (deterministic top candidate)")

    # 2. Temperature Sampling (temperature > 0.0)
    # Balanced distribution between two tokens
    balanced_logits = np.zeros(512, dtype=np.float32)
    balanced_logits[10] = 5.0
    balanced_logits[20] = 5.0
    sampled_set = set()
    for _ in range(50):
        tok = engine.sample(balanced_logits, temperature=1.0)
        sampled_set.add(tok)
    assert 10 in sampled_set and 20 in sampled_set, (
        f"Expected both candidate tokens under temperature=1.0, got {sampled_set}"
    )
    print("[PASS] Temperature sampling verified (distribution coverage)")

    # 3. Top-p Nucleus Sampling
    # Top-p of 0.1 with dominant token 42 should strictly isolate token 42
    top_p_samples = [engine.sample(logits_mx, temperature=0.5, top_p=0.1) for _ in range(20)]
    assert all(t == 42 for t in top_p_samples), f"Top-p isolation failed: {top_p_samples}"
    print("[PASS] Top-p nucleus sampling verified")


def test_reset_and_subsequent_generation():
    """Verify cache reset and reproducible subsequent generation."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    prompt = [1, 5, 9, 13, 17]

    # Run 1: initial generation
    run1_tokens = engine.generate(prompt, max_new_tokens=8, temperature=0.0)

    # Reset engine
    engine.reset()
    assert engine.sequence_length == 0, f"Expected sequence_length=0, got {engine.sequence_length}"
    for idx, layer in enumerate(engine.model.layers):
        assert layer.kv_cache.offset == 0, f"Layer {idx} cache offset not reset: {layer.kv_cache.offset}"

    # Run 2: identical prompt after reset must produce identical tokens under greedy decoding
    run2_tokens = engine.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert run1_tokens == run2_tokens, (
        f"Reproducibility failure after reset: Run 1 {run1_tokens} != Run 2 {run2_tokens}"
    )

    # Run 3: subsequent generation with a different prompt
    alt_prompt = [99, 100, 101]
    run3_tokens = engine.generate(alt_prompt, max_new_tokens=5, temperature=0.0)
    assert len(run3_tokens) == 5
    assert engine.sequence_length == len(alt_prompt) + 5 - 1

    print(f"[PASS] Engine reset and subsequent generation verified (reproduced {run1_tokens})")


def test_streaming_callback():
    """Verify that streaming callback is invoked for every generated token in real time."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    streamed_tokens = []

    def stream_cb(tok: int) -> None:
        assert isinstance(tok, int)
        streamed_tokens.append(tok)

    prompt = [7, 14, 21, 28]
    max_new_tokens = 6
    gen_tokens = engine.generate(
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        callback=stream_cb,
    )

    assert len(streamed_tokens) == max_new_tokens, (
        f"Expected {max_new_tokens} callback invocations, got {len(streamed_tokens)}"
    )
    assert streamed_tokens == gen_tokens, (
        f"Streamed tokens {streamed_tokens} do not match returned tokens {gen_tokens}"
    )
    print(f"[PASS] Streaming callback verified ({max_new_tokens} invocations in order)")


def test_mlx_backend_e2e():
    """Verify end-to-end prefill and decode execution under stock MLX baseline backend."""
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=256,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_dim=512,
        vocab_size=512,
        backend="mlx",
    )
    engine = InferenceEngine(cfg)

    prompt = [3, 6, 9, 12]
    tokens = engine.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert len(tokens) == 5
    assert engine.sequence_length == len(prompt) + 5 - 1
    print(f"[PASS] MLX baseline backend E2E generation verified ({tokens})")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TEST SUITE: UNIFIED ENGINE AND INFERENCE ORCHESTRATOR")
    print("=" * 80)

    test_engine_initialization()
    test_multi_token_prompt_prefill()
    test_autoregressive_generation_10_steps()
    test_sampling_modes()
    test_reset_and_subsequent_generation()
    test_streaming_callback()
    test_mlx_backend_e2e()

    print("=" * 80)
    print("[✓] ALL UNIFIED ENGINE ORCHESTRATOR TESTS PASSED CLEANLY (EXIT CODE 0)!")
    print("=" * 80)
