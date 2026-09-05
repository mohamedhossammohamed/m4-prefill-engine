"""
Real N-Gram / Prompt-Lookup Speculative Decoding (host-side decode loop).

This module implements PLD-style speculative decoding as PURE decode-loop
control above the existing forward pass. It touches no attention, MLP, GEMM,
or quantization-codec code -- zero kernel changes by construction.

Relation to the repo's existing "speculative" machinery (explicit comparison,
per directive -- nothing left running silently):
  * `streaming_1m_engine` Mode B (`executeModeBSpeculativeVerification`,
    K=64) is KERNEL-level attention verification over an out-of-core 1M KV
    stream. It assumes an EXTERNAL draft model proposes candidates and verifies
    them in one flash read. It is not wired into `InferenceEngine` and is
    untouched by this change (deferred experimental prototype).
  * This module is INFERENCE-LOOP prompt lookup: the draft comes from the
    already-seen text itself (prompt + generated so far), needs no draft
    model, and plugs into `InferenceEngine.generate_ngram_speculative`.
    The two occupy disjoint layers and never both run.

Losslessness: for greedy decoding (temperature=0) acceptance on argmax
equality is exactly lossless -- accepted tokens equal single-token greedy
output by construction. For sampled decoding the same-sample fast path is
used (statistically consistent; distribution-exact rejection sampling is
out of scope and documented at the call site).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpeculativeStats:
    """Counters for one `generate_ngram_speculative` run."""

    drafts_proposed: int = 0
    draft_tokens_proposed: int = 0
    tokens_accepted: int = 0
    fallback_steps: int = 0
    spec_model_calls: int = 0  # decode-phase forward passes used
    tokens_generated: int = 0

    @property
    def accept_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.tokens_accepted / self.draft_tokens_proposed

    @property
    def baseline_model_calls(self) -> int:
        """Sequential decoding would need one forward pass per token."""
        return self.tokens_generated

    @property
    def forward_passes_saved(self) -> int:
        return max(0, self.baseline_model_calls - self.spec_model_calls)

    def as_dict(self) -> dict:
        return {
            "drafts_proposed": self.drafts_proposed,
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "tokens_accepted": self.tokens_accepted,
            "fallback_steps": self.fallback_steps,
            "spec_model_calls": self.spec_model_calls,
            "tokens_generated": self.tokens_generated,
            "accept_rate": self.accept_rate,
            "baseline_model_calls": self.baseline_model_calls,
            "forward_passes_saved": self.forward_passes_saved,
        }


class NGramDrafter:
    """
    Prompt-lookup drafter: drafts K tokens by finding the most recent
    occurrence of the trailing N-token window in the already-seen text.
    """

    def __init__(self, n: int = 3, k: int = 4) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.n = int(n)
        self.k = int(k)

    def propose(self, context: List[int]) -> List[int]:
        """
        Returns up to K draft tokens following the most recent earlier
        occurrence of the trailing N-gram. Returns [] when no match exists
        or no tokens follow the match (graceful fallback signal).
        """
        n, k = self.n, self.k
        if len(context) < n:
            return []
        window = context[-n:]
        # Search earlier positions, most recent first. The window itself
        # starts at len(context)-n and cannot draft from itself.
        for start in range(len(context) - n - 1, -1, -1):
            if context[start : start + n] == window:
                following = context[start + n : start + n + k]
                if following:
                    return list(following)
                return []
        return []
