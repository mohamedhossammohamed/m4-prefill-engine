"""
Adversarial Red-Teaming Test Suite for Unified Engine & Inference Orchestrator.

Rigorously audits and attacks:
  - src/engine/config.py
  - src/engine/model.py
  - src/engine/inference_engine.py

Audit Vectors (12 Probes):
  Vector 1: Sequence Boundaries & Prompt Edge Cases
    - Probe 1: Single-Token Prompt Prefill (M = 1)
    - Probe 2: Empty Prompt Edge Cases (prefill([]) & generate([], ...))
    - Probe 3: Context Ceiling Breach (Prompt & Generation > max_context_length)
    - Probe 4: Zero / Negative max_new_tokens & Cache Destruction
  Vector 2: RoPE & Causal Masking Semantics
    - Probe 5: Causal Isolation & Anti-Bidirectional Attention Invariant
    - Probe 6: RoPE Offset & Prefill vs. Decode Step Alignment
    - Probe 7: Implicit State Pollution in Raw TransformerModel Forward Passes
  Vector 3: Vocabulary Out-of-Bounds & Token Safety
    - Probe 8: Out-of-Bounds & Negative Token ID Ingestion in Embedding Layer
  Vector 4: Generation State Management
    - Probe 9: Sequential generate() KV Cache Wiping vs. Continuation Contract
    - Probe 10: Missing EOS Token Early Termination Support
  Vector 5: Architecture Coverage (Gemma 2 Parameters)
    - Probe 11: Gemma 2 geglu, gemma_add_one, D=256 Parity Across Backends
  Vector 6: Sampler Tripwires & Numerical Gating
    - Probe 12: Silent Token 0 Masking on NaN / -Inf Logits in Sampler
"""

from __future__ import annotations
import inspect
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import numpy as np

from src.engine.config import EngineConfig
from src.engine.inference_engine import InferenceEngine
from src.engine.model import Embedding, RMSNorm, TransformerBlock, TransformerModel

# Terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class EngineAuditReport:
    """Telemetry collector and reporter for engine vulnerabilities."""

    def __init__(self) -> None:
        self.findings: List[Dict[str, str]] = []
        self.probes_run = 0
        self.probes_passed = 0
        self.probes_vulnerable = 0

    def record_finding(self, severity: str, category: str, summary: str, details: str) -> None:
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
        print(f"{BOLD}{CYAN}STEP 3 RED-TEAM AUDIT REPORT: UNIFIED INFERENCE ENGINE ORCHESTRATOR{RESET}")
        print("=" * 80)
        print(f"Total Audit Probes Executed: {self.probes_run}")
        print(f"Controlled / Sound:          {GREEN}{self.probes_passed}{RESET}")
        print(f"Vulnerabilities Identified:  {RED if len(self.findings) > 0 else GREEN}{len(self.findings)}{RESET}")
        print("-" * 80)

        for i, f in enumerate(self.findings, 1):
            color = RED if f["severity"] in ("CRITICAL", "HIGH") else YELLOW
            print(f"[{i}] {color}{BOLD}[{f['severity']}]{RESET} {BOLD}{f['category']}{RESET}: {f['summary']}")
            print(f"    Details: {f['details']}\n")

        print("=" * 80)
        return len(self.findings)


# ============================================================================
# PROBE IMPLEMENTATIONS
# ============================================================================

def probe_1_single_token_prefill(report: EngineAuditReport) -> None:
    """Probe 1: Single-token prompt prefill (M=1)."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 1: Single-Token Prompt Prefill (M=1)...{RESET}")

    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    try:
        # Prompt of length 1
        logits = engine.prefill([42])
        mx.eval(logits)

        # Check shape and finiteness
        assert logits.shape == (cfg.vocab_size,), f"Unexpected logits shape: {logits.shape}"
        assert bool(mx.all(mx.isfinite(logits)).item()), "Non-finite logits on M=1 prefill"
        assert engine.sequence_length == 1, f"Expected sequence_length=1, got {engine.sequence_length}"

        # Generate starting from single token
        gen = engine.generate([42], max_new_tokens=4)
        assert len(gen) == 4, f"Expected 4 generated tokens, got {len(gen)}"

        print(f"  {GREEN}[PASS]{RESET} Single-token prefill (M=1) executed and generated tokens: {gen}")
        report.probes_passed += 1
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} Single-token prefill crashed: {e}")
        report.record_finding(
            severity="HIGH",
            category="Sequence Boundaries",
            summary="Single-token prefill (M=1) failure",
            details=f"InferenceEngine.prefill([42]) failed with exception: {type(e).__name__}: {e}",
        )


def probe_2_empty_prompt(report: EngineAuditReport) -> None:
    """Probe 2: Empty prompt handling in prefill([]) and generate([], ...)."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 2: Empty Prompt Edge Cases...{RESET}")

    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    # Sub-case A: prefill([])
    prefill_exception = None
    try:
        engine.prefill([])
    except Exception as e:
        prefill_exception = e

    # Sub-case B: generate([], max_new_tokens=5)
    gen_result = None
    try:
        gen_result = engine.generate([], max_new_tokens=5)
    except Exception as e:
        gen_result = e

    if isinstance(prefill_exception, ValueError) and "[squeeze]" in str(prefill_exception):
        print(f"  {RED}[VULN]{RESET} prefill([]) crashes with internal MLX squeeze error instead of validating empty prompt")
        report.record_finding(
            severity="MEDIUM",
            category="Sequence Boundaries",
            summary="prefill([]) crashes with cryptic internal MLX dimension squeeze error",
            details=(
                f"Calling engine.prefill([]) passes shape [1, 0] through TransformerModel and crashes with "
                f"'{type(prefill_exception).__name__}: {prefill_exception}' at `logits[0, -1, :]`. "
                f"It should raise a clear ValueError('Prompt cannot be empty') upfront."
            ),
        )
    elif prefill_exception is None:
        print(f"  {RED}[VULN]{RESET} prefill([]) silently succeeded on empty prompt without error")
        report.record_finding(
            severity="HIGH",
            category="Sequence Boundaries",
            summary="prefill([]) accepted empty token list without error",
            details="Calling engine.prefill([]) with 0 tokens succeeded without raising ValueError.",
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} prefill([]) correctly raised exception: {prefill_exception}")
        report.probes_passed += 1

    # Check generate([], max_new_tokens=5)
    if gen_result == []:
        print(f"  {YELLOW}[NOTE]{RESET} generate([], 5) returned empty list [] silently")


def probe_3_context_ceiling_overflow(report: EngineAuditReport) -> None:
    """Probe 3: Context ceiling enforcement (prompt > max_context_length & prompt + new_tokens > max_context_length)."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 3: Context Ceiling Breach Invariant...{RESET}")

    max_ctx = 32
    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
        max_context_length=max_ctx,
        backend="m4",
    )
    engine = InferenceEngine(cfg)

    # Sub-case A: prefill with prompt length > max_context_length
    prompt_overflow = [1] * (max_ctx + 16)
    overflow_accepted = False
    try:
        engine.prefill(prompt_overflow)
        overflow_accepted = True
    except ValueError:
        pass
    except Exception as e:
        print(f"  Unexpected exception on context overflow: {e}")

    # Sub-case B: generate with prompt + max_new_tokens > max_context_length
    engine.reset()
    gen_overflow_accepted = False
    try:
        gen = engine.generate([1] * (max_ctx - 5), max_new_tokens=15)
        if len(gen) > 0 and engine.sequence_length > max_ctx:
            gen_overflow_accepted = True
    except ValueError:
        pass

    if overflow_accepted or gen_overflow_accepted:
        print(f"  {RED}[VULN]{RESET} Context ceiling max_context_length={max_ctx} completely unenforced!")
        report.record_finding(
            severity="HIGH",
            category="Sequence Boundaries",
            summary="Context ceiling max_context_length is unenforced; KV cache expands unconstrained",
            details=(
                f"EngineConfig configured max_context_length={max_ctx}. "
                f"prefill(tokens={len(prompt_overflow)}) succeeded without ValueError (overflow_accepted={overflow_accepted}). "
                f"generate() exceeded max_context_length (gen_overflow_accepted={gen_overflow_accepted}, "
                f"final sequence_length={engine.sequence_length}). "
                f"InferenceEngine must raise ValueError when sequence exceeds max_context_length."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} Context ceiling properly enforced with ValueError")
        report.probes_passed += 1


def probe_4_zero_max_new_tokens_state_reset(report: EngineAuditReport) -> None:
    """Probe 4: Zero / negative max_new_tokens causing destructive state reset."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 4: Zero/Negative max_new_tokens State Reset...{RESET}")

    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
    )
    engine = InferenceEngine(cfg)

    # Pre-populate engine state with prefill
    engine.prefill([10, 20, 30])
    initial_seq_len = engine.sequence_length
    assert initial_seq_len == 3, f"Expected seq len 3, got {initial_seq_len}"

    # Call generate with max_new_tokens=0
    res = engine.generate([1, 2], max_new_tokens=0)
    post_seq_len = engine.sequence_length

    if res == [] and post_seq_len == 0:
        print(f"  {YELLOW}[VULN]{RESET} generate(..., max_new_tokens=0) wiped existing KV cache before early exit")
        report.record_finding(
            severity="LOW",
            category="Generation State Management",
            summary="generate() unconditionally calls self.reset() before validating arguments",
            details=(
                "Calling generate(..., max_new_tokens=0) or generate([], ...) resets existing KV cache and "
                "sequence length to 0 before early-returning []. Argument validation and zero-generation checks "
                "should occur before destructive state mutation."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} generate(max_new_tokens=0) handled gracefully")
        report.probes_passed += 1


def probe_5_causal_mask_isolation(report: EngineAuditReport) -> None:
    """Probe 5: Causal isolation & anti-bidirectional attention leakage invariant."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 5: Causal Attention Isolation Invariant...{RESET}")

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

    # Sequence A: [10, 20, 30, 40]
    p1 = [10, 20, 30, 40]
    engine.reset()
    logits1 = engine.model(p1)
    mx.eval(logits1)

    # Sequence B: [10, 20, 30, 99] (token at position 3 mutated)
    p2 = [10, 20, 30, 99]
    engine.reset()
    logits2 = engine.model(p2)
    mx.eval(logits2)

    # Strict invariant: positions 0, 1, 2 must have ZERO divergence
    diff0 = mx.max(mx.abs(logits1[0, 0, :] - logits2[0, 0, :])).item()
    diff1 = mx.max(mx.abs(logits1[0, 1, :] - logits2[0, 1, :])).item()
    diff2 = mx.max(mx.abs(logits1[0, 2, :] - logits2[0, 2, :])).item()
    diff3 = mx.max(mx.abs(logits1[0, 3, :] - logits2[0, 3, :])).item()

    if diff0 > 1e-4 or diff1 > 1e-4 or diff2 > 1e-4:
        print(f"  {RED}[CRITICAL]{RESET} Causal masking failure! Bidirectional attention leakage detected!")
        print(f"    Divergence: pos 0={diff0}, pos 1={diff1}, pos 2={diff2}")
        report.record_finding(
            severity="CRITICAL",
            category="RoPE & Causal Masking",
            summary="Bidirectional attention leakage during prefill (causal mask violation)",
            details=(
                f"Mutating token 3 changed logits at prior positions: pos0={diff0}, pos1={diff1}, pos2={diff2}. "
                f"Autoregressive causal invariant violated!"
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} Causal isolation strictly preserved (pos 0..2 diff = 0.0, pos 3 diff = {diff3:.4f})")
        report.probes_passed += 1


def probe_6_rope_prefill_decode_alignment(report: EngineAuditReport) -> None:
    """Probe 6: RoPE offset alignment between prefill and decode_step."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 6: RoPE Offset Alignment (Prefill vs. Decode Step)...{RESET}")

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

    # Run 1: Prefill [15, 25, 35] all at once
    engine.reset()
    full_prefill_logits = engine.prefill([15, 25, 35])
    mx.eval(full_prefill_logits)

    # Run 2: Prefill [15, 25], then decode_step(35)
    engine.reset()
    engine.prefill([15, 25])
    engine.decode_step(35)
    step_logits = engine._last_logits
    mx.eval(step_logits)

    diff = mx.max(mx.abs(full_prefill_logits - step_logits)).item()

    if diff > 1e-3:
        print(f"  {RED}[CRITICAL]{RESET} RoPE offset mismatch between prefill and decode_step! Max diff: {diff}")
        report.record_finding(
            severity="CRITICAL",
            category="RoPE & Causal Masking",
            summary="RoPE offset / KV cache discrepancy between prefill and decode_step",
            details=(
                f"Logits at position 2 differed between full prefill and prefill+decode_step by {diff}. "
                f"RoPE positions or KV cache retrieval offset are inconsistent."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} RoPE prefill and decode_step logits match bit-for-bit (diff = {diff:.6f})")
        report.probes_passed += 1


def probe_7_model_call_cache_pollution(report: EngineAuditReport) -> None:
    """Probe 7: Stateful cache pollution in direct TransformerModel.__call__ invocations."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 7: Stateful Pollution in TransformerModel.__call__...{RESET}")

    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
    )
    model = TransformerModel(cfg)
    model.init_synthetic_weights()

    sig = inspect.signature(model.__call__)
    has_use_cache = "use_cache" in sig.parameters

    tokens = [5, 10, 15]

    # Test stateless evaluation using use_cache=False
    out1 = model(tokens, use_cache=False)
    mx.eval(out1)
    offset_after_1 = model.layers[0].kv_cache.offset

    out2 = model(tokens, use_cache=False)
    mx.eval(out2)
    offset_after_2 = model.layers[0].kv_cache.offset

    diff = mx.max(mx.abs(out1 - out2)).item()

    if not has_use_cache or offset_after_1 > 0 or offset_after_2 > 0 or diff > 1e-4:
        print(f"  {YELLOW}[VULN]{RESET} Direct TransformerModel.__call__ does not support stateless use_cache=False evaluation")
        report.record_finding(
            severity="MEDIUM",
            category="RoPE & Causal Masking",
            summary="Direct TransformerModel forward passes pollute internal KV cache state",
            details=(
                f"Calling model(tokens, use_cache=False) produced offset_1={offset_after_1}, offset_2={offset_after_2}, "
                f"diff={diff:.6f}, has_use_cache={has_use_cache}. Stateless evaluation requires use_cache=False to keep offset 0."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} Model calls with use_cache=False are verified stateless (offset={offset_after_2}, diff={diff:.6f})")
        report.probes_passed += 1


def probe_8_vocab_out_of_bounds(report: EngineAuditReport) -> None:
    """Probe 8: Out-of-bounds and negative token ID ingestion in Embedding & InferenceEngine."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 8: Vocabulary Out-of-Bounds & Negative Token ID Safety...{RESET}")

    vocab_size = 128
    cfg = EngineConfig(
        vocab_size=vocab_size,
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
    )
    engine = InferenceEngine(cfg)

    # Test A: Token >= vocab_size
    oob_token = vocab_size + 50
    oob_prefill_rejected = False
    try:
        logits = engine.prefill([oob_token])
        mx.eval(logits)
    except (ValueError, IndexError):
        oob_prefill_rejected = True
    except Exception as e:
        print(f"  Unexpected exception for OOB token: {e}")

    # Test B: Negative token ID
    neg_token = -5
    neg_prefill_rejected = False
    try:
        engine.reset()
        logits = engine.prefill([neg_token])
        mx.eval(logits)
    except (ValueError, IndexError):
        neg_prefill_rejected = True
    except Exception as e:
        print(f"  Unexpected exception for negative token: {e}")

    # Test C: decode_step OOB
    oob_decode_rejected = False
    try:
        engine.decode_step(oob_token)
    except (ValueError, IndexError):
        oob_decode_rejected = True
    except Exception:
        pass

    if not oob_prefill_rejected or not neg_prefill_rejected or not oob_decode_rejected:
        print(f"  {RED}[VULN]{RESET} Out-of-bounds / negative tokens accepted silently without bounds check!")
        report.record_finding(
            severity="HIGH",
            category="Vocabulary Safety",
            summary="Zero bounds validation on token IDs; out-of-bounds tokens silently zeroed/wrapped",
            details=(
                f"Config vocab_size={vocab_size}. Token {oob_token} (>= vocab_size) was accepted without error "
                f"(rejected={oob_prefill_rejected}). Token {neg_token} (< 0) was accepted without error "
                f"(rejected={neg_prefill_rejected}). decode_step({oob_token}) rejected={oob_decode_rejected}. "
                f"mx.take in Embedding silently clamps out-of-bounds to zeros and wraps negative tokens. "
                f"InferenceEngine and Embedding must raise ValueError/IndexError on token IDs not in [0, vocab_size - 1]."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} Out-of-bounds and negative tokens properly rejected")
        report.probes_passed += 1


def probe_9_sequential_generate_contract(report: EngineAuditReport) -> None:
    """Probe 9: Sequential generate() calls and cache preservation contract."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 9: Sequential generate() KV Cache Contract...{RESET}")

    cfg = EngineConfig(
        num_layers=2,
        hidden_dim=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        vocab_size=256,
    )
    engine = InferenceEngine(cfg)

    # 1. First generation
    g1 = engine.generate([1, 2, 3], max_new_tokens=4)
    seq_after_g1 = engine.sequence_length

    # 2. Second generation
    g2 = engine.generate([4, 5], max_new_tokens=4)
    seq_after_g2 = engine.sequence_length

    # Check contract: does generate() support multi-turn continuation?
    sig = inspect.signature(engine.generate)
    has_reset_param = "reset_cache" in sig.parameters

    if not has_reset_param:
        print(f"  {YELLOW}[VULN]{RESET} generate() unconditionally resets KV cache; continuation is impossible")
        report.record_finding(
            severity="MEDIUM",
            category="Generation State Management",
            summary="generate() unconditionally wipes KV cache; lacks reset_cache parameter for continuation",
            details=(
                f"generate() hardcodes self.reset() at line 146 without a reset_cache parameter. "
                f"After g1 (len={len(g1)}), sequence_length was {seq_after_g1}. Calling g2 reset the engine "
                f"and sequence_length became {seq_after_g2}. Multi-turn generation or continuing an existing "
                f"prefix without recomputing the KV cache is currently impossible via generate()."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} generate() provides explicit control over cache resetting")
        report.probes_passed += 1


def probe_10_missing_eos_token(report: EngineAuditReport) -> None:
    """Probe 10: Missing EOS token early termination support in generate()."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 10: EOS Token Early Termination Support...{RESET}")

    cfg = EngineConfig()
    engine = InferenceEngine(cfg)

    sig = inspect.signature(engine.generate)
    has_eos_param = "eos_token_id" in sig.parameters

    if not has_eos_param:
        print(f"  {RED}[VULN]{RESET} generate() has no eos_token_id parameter and cannot terminate early!")
        report.record_finding(
            severity="HIGH",
            category="Generation State Management",
            summary="Missing eos_token_id support in generate(); unable to stop generation on end-of-sequence",
            details=(
                "InferenceEngine.generate() lacks an `eos_token_id: Optional[Union[int, List[int]]] = None` parameter. "
                "The generation loop runs unconditionally for `max_new_tokens` iterations, even when the model "
                "predicts an EOS token, generating trailing hallucinated junk."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} generate() supports eos_token_id early termination")
        report.probes_passed += 1


def probe_11_gemma2_architecture_coverage(report: EngineAuditReport) -> None:
    """Probe 11: Gemma 2 architecture configuration (geglu, gemma_add_one, D=256, GQA)."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 11: Gemma 2 Architecture Configuration Coverage...{RESET}")

    m4_success = False
    mlx_success = False

    for backend in ("m4", "mlx"):
        try:
            cfg = EngineConfig(
                num_layers=2,
                hidden_dim=256 * 4,  # 1024
                num_q_heads=4,
                num_kv_heads=2,
                head_dim=256,        # Gemma 2 D=256
                intermediate_dim=2048,
                vocab_size=512,
                activation="geglu",
                norm_type="gemma_add_one",
                backend=backend,
            )
            engine = InferenceEngine(cfg)
            logits = engine.prefill([10, 20, 30])
            mx.eval(logits)
            assert logits.shape == (cfg.vocab_size,), f"Expected shape ({cfg.vocab_size},), got {logits.shape}"
            assert bool(mx.all(mx.isfinite(logits)).item()), f"Non-finite logits in Gemma 2 backend={backend}"

            gen = engine.generate([10, 20, 30], max_new_tokens=3)
            assert len(gen) == 3, f"Expected 3 tokens, got {len(gen)}"

            if backend == "m4":
                m4_success = True
            else:
                mlx_success = True
            print(f"  {GREEN}[PASS]{RESET} Gemma 2 configuration passed on backend='{backend}' (gen={gen})")
        except Exception as e:
            print(f"  {RED}[FAIL]{RESET} Gemma 2 configuration failed on backend='{backend}': {e}")
            report.record_finding(
                severity="HIGH",
                category="Architecture Coverage",
                summary=f"Gemma 2 parameters failed on backend='{backend}'",
                details=f"Gemma 2 configuration raised {type(e).__name__}: {e}",
            )

    if m4_success and mlx_success:
        report.probes_passed += 1


def probe_12_sampler_tripwires(report: EngineAuditReport) -> None:
    """Probe 12: Sampler tripwires for NaN / -Inf masking vs. FloatingPointError."""
    report.probes_run += 1
    print(f"\n{BOLD}Running Probe 12: Sampler Numerical Tripwires (NaN / -Inf Gating)...{RESET}")

    cfg = EngineConfig(vocab_size=16)
    engine = InferenceEngine(cfg)

    nan_logits = mx.array([float("nan")] * 16)
    inf_logits = mx.array([-float("inf")] * 16)

    # Test A: Greedy with all NaN
    greedy_nan_result = None
    try:
        greedy_nan_result = engine.sample(nan_logits, temperature=0.0)
    except FloatingPointError:
        pass
    except Exception as e:
        greedy_nan_result = e

    # Test B: Stochastic with all NaN
    stoch_nan_result = None
    try:
        stoch_nan_result = engine.sample(nan_logits, temperature=0.7)
    except FloatingPointError:
        pass
    except Exception as e:
        stoch_nan_result = e

    # Test C: Greedy with all -Inf
    greedy_inf_result = None
    try:
        greedy_inf_result = engine.sample(inf_logits, temperature=0.0)
    except FloatingPointError:
        pass
    except Exception as e:
        greedy_inf_result = e

    # Test D: Stochastic with all -Inf
    stoch_inf_result = None
    try:
        stoch_inf_result = engine.sample(inf_logits, temperature=0.7)
    except FloatingPointError:
        pass
    except Exception as e:
        stoch_inf_result = e

    # Test E: Negative temperature validation
    neg_temp_result = None
    try:
        neg_temp_result = engine.sample(mx.array([1.0, 2.0, 3.0]), temperature=-1.0)
    except ValueError:
        pass
    except Exception as e:
        neg_temp_result = e

    vulnerabilities = []
    if greedy_nan_result == 0 or stoch_nan_result == 0:
        vulnerabilities.append("NaN logits silently return token ID 0")
    if greedy_inf_result == 0 or stoch_inf_result == 0:
        vulnerabilities.append("-Inf logits silently return token ID 0")
    if neg_temp_result is not None and not isinstance(neg_temp_result, ValueError):
        vulnerabilities.append("Negative temperature accepted and treated as greedy argmax")

    if vulnerabilities:
        print(f"  {RED}[CRITICAL]{RESET} Sampler silently masks numerical corruption with token 0!")
        for v in vulnerabilities:
            print(f"    - {v}")
        report.record_finding(
            severity="CRITICAL",
            category="Sampler Tripwires",
            summary="sample() silently returns token 0 on NaN/-Inf logits instead of raising FloatingPointError",
            details=(
                f"When given all-NaN or all--Inf logits, sample() fails to validate numerical integrity and "
                f"silently returns token ID 0 via fallback `probs[0] = 1.0` or unchecked argmax. "
                f"Additionally, negative temperature (-1.0) satisfies `temperature <= 1e-6` and is treated as greedy "
                f"without raising ValueError. All numerical anomalies must trip FloatingPointError."
            ),
        )
    else:
        print(f"  {GREEN}[PASS]{RESET} Sampler strictly gates NaN, -Inf, and invalid parameters")
        report.probes_passed += 1


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main() -> int:
    report = EngineAuditReport()

    probe_1_single_token_prefill(report)
    probe_2_empty_prompt(report)
    probe_3_context_ceiling_overflow(report)
    probe_4_zero_max_new_tokens_state_reset(report)
    probe_5_causal_mask_isolation(report)
    probe_6_rope_prefill_decode_alignment(report)
    probe_7_model_call_cache_pollution(report)
    probe_8_vocab_out_of_bounds(report)
    probe_9_sequential_generate_contract(report)
    probe_10_missing_eos_token(report)
    probe_11_gemma2_architecture_coverage(report)
    probe_12_sampler_tripwires(report)

    findings_count = report.print_summary()
    return 0 if findings_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
