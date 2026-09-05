"""
Verification suite for real N-gram / prompt-lookup speculative decoding.

  (a) correctness: greedy speculative output identical to single-token
      generation (lossless by construction) -- real tiny model + fake models.
  (b) repetitive-text: guaranteed n-gram matches -> accept_rate > 0 and
      strictly fewer forward passes than sequential (FakeCyclicModel).
  (c) no-repetition fallback: identical output, no crash (FakeCounterModel
      + real model on random prompt).

Zero kernel changes: this exercises only the host decode loop.
Run: .venv/bin/python tests/test_ngram_speculative.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np

from src.engine import EngineConfig, InferenceEngine
from src.engine.ngram_drafter import NGramDrafter, SpeculativeStats


class FakeTransitionModel:
    """Pure function of input content (batching-independent, like a real model).

    Position j of any forward call predicts `transition(input[j])`, so
    sequential and batched verification always agree: losslessness holds by
    construction regardless of call pattern.
    """

    def __init__(self, vocab_size: int, transition):
        self.vocab_size = vocab_size
        self.transition = transition
        self.calls = 0

    def reset(self):
        self.calls = 0

    def __call__(self, tokens, mask=None, use_cache=True):
        arr = mx.array(tokens, dtype=mx.int32) if not isinstance(tokens, mx.array) else tokens
        B, M = arr.shape[0], arr.shape[1]
        flat = np.array(arr.reshape(-1)).astype(int).tolist()
        out = np.zeros((B, M, self.vocab_size), dtype=np.float32)
        for i, tok in enumerate(flat):
            out[:, i, self.transition(int(tok)) % self.vocab_size] = 10.0
        self.calls += 1
        return mx.array(out)


def _cyclic_transition(a, b, c):
    table = {a: b, b: c, c: a}
    return lambda x: table.get(x, (x * 7 + 1) % 1024)


def _tiny_cfg(**kw):
    base = dict(
        num_layers=2, hidden_dim=128, num_q_heads=4, num_kv_heads=2,
        head_dim=32, intermediate_dim=256, vocab_size=1024,
        backend="mlx", max_context_length=512,
    )
    base.update(kw)
    return EngineConfig(**base)


def test_drafter_unit():
    d = NGramDrafter(n=3, k=4)
    # Most-recent earlier match wins: window [2,3,4] earlier at index 1
    # -> following K=4 tokens [5,6,2,3].
    assert d.propose([1, 2, 3, 4, 5, 6, 2, 3, 4]) == [5, 6, 2, 3]
    # No match -> graceful empty.
    assert d.propose([11, 22, 33, 44]) == []
    # Match at tail with nothing following -> empty (no self-draft).
    assert d.propose([9, 9, 9]) == []
    # Context shorter than N -> empty.
    assert d.propose([1, 2]) == []
    # K clipping.
    d2 = NGramDrafter(n=2, k=2)
    assert d2.propose([5, 6, 7, 8, 9, 5, 6]) == [7, 8]
    print("[PASS] NGramDrafter unit semantics (recent-match, fallback, K-clip)")


def test_lossless_real_model():
    cfg = _tiny_cfg()
    prompt = [1, 2, 3, 1, 2, 3, 7, 8]
    e1 = InferenceEngine(cfg)
    base = e1.generate(prompt, max_new_tokens=12, temperature=0.0)
    e2 = InferenceEngine(cfg)
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=12, temperature=0.0)
    assert base == spec, f"LOSSLESS VIOLATION: {base} != {spec}"
    print(f"[PASS] Real-model greedy losslessness ({spec})")


def test_repetitive_accept_and_speedup():
    cfg = _tiny_cfg()
    pattern = [21, 22, 23]
    prompt = pattern * 3  # [21,22,23,21,22,23,21,22,23]
    trans = _cyclic_transition(*pattern)
    e1 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    base = e1.generate(prompt, max_new_tokens=9, temperature=0.0)
    e2 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=9, temperature=0.0)
    assert base == spec, f"LOSSLESS VIOLATION: {base} != {spec}"
    assert stats.drafts_proposed > 0, "expected drafts on repetitive text"
    assert stats.accept_rate > 0.0, f"expected nonzero accept-rate, got {stats.as_dict()}"
    assert stats.forward_passes_saved > 0, f"expected saved passes, got {stats.as_dict()}"
    # Sequential baseline decode-phase calls for 9 tokens: 1 prefill-excluded => 8 decode calls.
    assert stats.spec_model_calls < 8, f"expected fewer calls, got {stats.as_dict()}"
    print(f"[PASS] Repetitive accept+speedup (accept_rate={stats.accept_rate:.2f}, "
          f"calls={stats.spec_model_calls} vs baseline={stats.baseline_model_calls})")


def test_no_repetition_fallback():
    cfg = _tiny_cfg()
    prompt = [101, 202, 303, 404]
    trans = lambda x: (x + 1) % cfg.vocab_size  # outputs never repeat history
    e1 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    base = e1.generate(prompt, max_new_tokens=8, temperature=0.0)
    e2 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=8, temperature=0.0)
    assert base == spec, f"FALLBACK CORRECTNESS VIOLATION: {base} != {spec}"
    assert len(spec) == 8 and all(0 <= t < cfg.vocab_size for t in spec)
    print(f"[PASS] No-repetition fallback identical + no crash ({stats.as_dict()})")


def test_mismatch_truncation_path():
    # Drafts proposed but model disagrees at first position: exercises the
    # reject-and-truncate branch while staying bit-identical to baseline.
    cfg = _tiny_cfg()
    # Window [1,2,3] matches history (draft [4,1,2]) but T(4)=5 != 1:
    # mismatch at position 1 exercises reject-and-truncate.
    prompt = [1, 2, 3, 4, 1, 2]
    trans = lambda x: (x + 1) % cfg.vocab_size
    e1 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    base = e1.generate(prompt, max_new_tokens=8, temperature=0.0)
    e2 = InferenceEngine(cfg, model=FakeTransitionModel(cfg.vocab_size, trans))
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=8, temperature=0.0)
    assert base == spec, f"MISMATCH-PATH VIOLATION: {base} != {spec}"
    assert stats.drafts_proposed > 0, f"expected proposals, got {stats.as_dict()}"
    print(f"[PASS] Mismatch truncation identical ({stats.as_dict()})")


def test_real_model_random_prompt_fallback():
    cfg = _tiny_cfg()
    prompt = [913, 17, 555, 42]
    e1 = InferenceEngine(cfg)
    base = e1.generate(prompt, max_new_tokens=10, temperature=0.0)
    e2 = InferenceEngine(cfg)
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=10, temperature=0.0)
    assert base == spec, f"REAL-MODEL FALLBACK VIOLATION: {base} != {spec}"
    print(f"[PASS] Real-model random-prompt fallback identical (accept_rate={stats.accept_rate:.2f})")


def test_m4_backend_parity():
    cfg = _tiny_cfg(backend="m4")
    prompt = [4, 8, 15, 16, 23, 42]
    e1 = InferenceEngine(cfg)
    base = e1.generate(prompt, max_new_tokens=8, temperature=0.0)
    e2 = InferenceEngine(cfg)
    spec, stats = e2.generate_ngram_speculative(prompt, max_new_tokens=8, temperature=0.0)
    assert base == spec, f"M4 PARITY VIOLATION: {base} != {spec}"
    print(f"[PASS] M4-backend speculative parity ({spec})")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TEST SUITE: N-GRAM SPECULATIVE DECODING (Task 2)")
    print("=" * 80)
    test_drafter_unit()
    test_lossless_real_model()
    test_repetitive_accept_and_speedup()
    test_no_repetition_fallback()
    test_mismatch_truncation_path()
    test_real_model_random_prompt_fallback()
    test_m4_backend_parity()
    print("=" * 80)
    print("ALL N-GRAM SPECULATIVE TESTS PASSED (EXIT CODE 0)")
    print("=" * 80)
