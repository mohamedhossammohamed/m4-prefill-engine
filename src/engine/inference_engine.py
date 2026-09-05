"""
Unified Inference Engine for Prefill and Autoregressive Token Generation.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Union

import mlx.core as mx
import numpy as np

from core.bridge.m4_bridge import MetalUMABridge, m4_bridge_synchronize
from src.engine.config import EngineConfig
from src.engine.model import TransformerModel
from src.engine.ngram_drafter import NGramDrafter, SpeculativeStats


class InferenceEngine:
    """
    Unified transformer inference orchestrator capable of prompt prefill
    and autoregressive generation on Apple Silicon unified memory.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model: Optional[TransformerModel] = None,
    ) -> None:
        if config is None:
            config = EngineConfig(backend="m4")
        self.config = config

        if model is None:
            model = TransformerModel(config)
            model.init_synthetic_weights()
        self.model = model

        self._sequence_length: int = 0
        self._last_token: Optional[int] = None
        self._last_logits: Optional[mx.array] = None

    @property
    def sequence_length(self) -> int:
        """Total tokens currently processed in the KV cache."""
        return self._sequence_length

    def sample(
        self,
        logits: Union[mx.array, np.ndarray],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """
        Samples a single token ID from unnormalized logits.
        Supports greedy argmax (temperature=0.0) and temperature + top_p sampling.
        """
        if temperature < 0.0:
            raise ValueError("Temperature must be non-negative.")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0.")

        if isinstance(logits, mx.array):
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise FloatingPointError("Tripwire: Non-finite values (NaN/Inf) detected in logits during sampling.")
        else:
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError("Tripwire: Non-finite values (NaN/Inf) detected in logits during sampling.")

        if temperature <= 1e-6:
            if isinstance(logits, mx.array):
                return int(mx.argmax(logits).item())
            return int(np.argmax(logits))

        if isinstance(logits, mx.array):
            logits_np = np.array(logits.astype(mx.float32))
        else:
            logits_np = np.array(logits, dtype=np.float32)

        scaled = logits_np / float(temperature)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs /= sum_probs
        else:
            probs = np.zeros_like(probs)
            probs[0] = 1.0

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cum_probs = np.cumsum(sorted_probs)
            cutoff_mask = cum_probs > top_p
            cutoff_mask[0] = False  # Keep at least the top candidate
            sorted_probs[cutoff_mask] = 0.0
            sum_p = np.sum(sorted_probs)
            if sum_p > 0:
                sorted_probs /= sum_p
            else:
                sorted_probs = np.zeros_like(sorted_probs)
                sorted_probs[0] = 1.0
            idx = np.random.choice(len(sorted_probs), p=sorted_probs)
            return int(sorted_indices[idx])

        return int(np.random.choice(len(probs), p=probs))

    def prefill(self, tokens: Union[List[int], mx.array]) -> mx.array:
        """
        Runs prompt prefill (M > 1), caches KV tokens, and returns final-step logits.
        """
        if isinstance(tokens, list):
            if len(tokens) == 0:
                raise ValueError("Prompt tokens cannot be empty for prefill.")
            tokens_arr = mx.array(tokens, dtype=mx.int32)
        else:
            if tokens.size == 0:
                raise ValueError("Prompt tokens cannot be empty for prefill.")
            tokens_arr = tokens.astype(mx.int32)

        if bool(mx.any(tokens_arr < 0).item()) or bool(mx.any(tokens_arr >= self.config.vocab_size).item()):
            raise IndexError(f"Token ID out of vocabulary bounds [0, {self.config.vocab_size})")

        if tokens_arr.ndim == 1:
            tokens_2d = tokens_arr[None, :]
        else:
            tokens_2d = tokens_arr

        M = tokens_2d.shape[1]
        if self._sequence_length + M > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({self._sequence_length + M}) exceeds max_context_length ({self.config.max_context_length})"
            )

        logits = self.model(tokens_2d)
        mx.eval(logits)
        m4_bridge_synchronize()
        self._sequence_length += M
        final_logits = logits[0, -1, :]
        self._last_logits = final_logits
        return final_logits

    def decode_step(
        self,
        next_token: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """
        Runs single autoregressive decode step (M = 1), appends to KV cache,
        and returns next sampled token ID.
        """
        if next_token < 0 or next_token >= self.config.vocab_size:
            raise IndexError(f"Token ID {next_token} out of vocabulary bounds [0, {self.config.vocab_size})")

        if self._sequence_length + 1 > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({self._sequence_length + 1}) exceeds max_context_length ({self.config.max_context_length})"
            )

        tok_arr = mx.array([[int(next_token)]], dtype=mx.int32)
        logits = self.model(tok_arr)
        mx.eval(logits)
        m4_bridge_synchronize()
        self._sequence_length += 1
        step_logits = logits[0, -1, :]
        self._last_logits = step_logits
        sampled_id = self.sample(step_logits, temperature=temperature, top_p=top_p)
        self._last_token = sampled_id
        return sampled_id

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 1.0,
        callback: Optional[Callable[[int], None]] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        reset_cache: bool = True,
    ) -> List[int]:
        """
        Full generation loop: runs prompt prefill, then decodes max_new_tokens autoregressively.
        """
        # Validate arguments first before state mutation
        if max_new_tokens <= 0 or len(prompt_tokens) == 0:
            return []

        start_len = 0 if reset_cache else self._sequence_length
        if start_len + len(prompt_tokens) > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({start_len + len(prompt_tokens)}) exceeds max_context_length ({self.config.max_context_length})"
            )
        if start_len + len(prompt_tokens) + max_new_tokens - 1 > self.config.max_context_length:
            raise ValueError(
                f"Total generation sequence length ({start_len + len(prompt_tokens) + max_new_tokens - 1}) exceeds max_context_length ({self.config.max_context_length})"
            )

        if reset_cache:
            self.reset()

        eos_set = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                eos_set = set(int(t) for t in eos_token_id)
            else:
                eos_set = {int(eos_token_id)}

        # 1. Prompt prefill
        final_logits = self.prefill(prompt_tokens)

        # 2. Sample first token from prompt's final logits
        first_token = self.sample(final_logits, temperature=temperature, top_p=top_p)
        generated: List[int] = [first_token]
        if callback is not None:
            callback(first_token)

        if first_token in eos_set:
            return generated

        # 3. Autoregressive decode steps
        curr_token = first_token
        for _ in range(1, max_new_tokens):
            next_tok = self.decode_step(curr_token, temperature=temperature, top_p=top_p)
            generated.append(next_tok)
            if callback is not None:
                callback(next_tok)
            if next_tok in eos_set:
                break
            curr_token = next_tok

        return generated

    def reset(self) -> None:
        """Resets KV cache and sequence counters."""
        self.model.reset()
        self._sequence_length = 0
        self._last_token = None
        self._last_logits = None

    # ------------------------------------------------------------------
    # N-gram speculative decoding (host-side only, zero kernel changes)
    # ------------------------------------------------------------------

    def _spec_caches_snapshot(self) -> Optional[dict]:
        """Snapshots per-layer KV buffers + offsets. None if model has none."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            return {"seq_len": self._sequence_length, "layers": []}
        snap_layers = []
        for layer in layers:
            cache = getattr(layer, "kv_cache", None)
            if cache is None:
                snap_layers.append(None)
                continue
            entry: dict = {"cache": cache, "offset": cache.offset}
            for attr in ("_k_buf", "_v_buf", "keys", "values"):
                buf = getattr(cache, attr, None)
                if buf is not None:
                    mx.eval(buf)
                    entry[attr] = mx.array(np.array(buf, copy=True))
                else:
                    entry[attr] = None
            for attr in ("_capacity", "_seq_dim", "_total_streamed_tokens"):
                if hasattr(cache, attr):
                    entry[attr] = getattr(cache, attr)
            snap_layers.append(entry)
        return {"seq_len": self._sequence_length, "layers": snap_layers}

    def _spec_caches_restore(self, snap: Optional[dict]) -> None:
        """Restores a snapshot taken by `_spec_caches_snapshot`."""
        if not snap:
            return
        self._sequence_length = snap["seq_len"]
        for entry in snap["layers"]:
            if entry is None:
                continue
            cache = entry["cache"]
            cache._offset = entry["offset"]
            for attr in ("_k_buf", "_v_buf", "keys", "values"):
                if attr in entry:
                    setattr(cache, attr, entry[attr])
            if "_capacity" in entry:
                cache._capacity = entry["_capacity"]
            if "_total_streamed_tokens" in entry:
                cache._total_streamed_tokens = entry["_total_streamed_tokens"]

    def _spec_caches_truncate(self, target_len: int) -> None:
        """Truncates live caches back to `target_len` tokens (post-reject)."""
        self._sequence_length = target_len
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        for layer in layers:
            cache = getattr(layer, "kv_cache", None)
            if cache is None:
                continue
            cache._offset = target_len
            # MLX concat-style caches: slice arrays; M4 preallocated
            # buffers are overwritten in place so offset reset suffices.
            for attr in ("keys", "values"):
                buf = getattr(cache, attr, None)
                if buf is not None and hasattr(buf, "ndim"):
                    if buf.ndim == 4:
                        seq = 2 if buf.shape[1] == getattr(cache, "n_heads", -1) else 1
                        if seq == 2:
                            setattr(cache, attr, buf[:, :, :target_len, :])
                        else:
                            setattr(cache, attr, buf[:, :target_len, :, :])
                    elif buf.ndim == 3:
                        setattr(cache, attr, buf[:, :target_len, :])

    def _spec_uses_out_of_core(self) -> bool:
        layers = getattr(self.model, "layers", None)
        if not layers:
            return False
        return any(getattr(getattr(l, "kv_cache", None), "mode", "in_ram") != "in_ram" for l in layers)

    def generate_ngram_speculative(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        n: int = 3,
        k: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        callback: Optional[Callable[[int], None]] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        reset_cache: bool = True,
    ) -> tuple:
        """
        N-gram prompt-lookup speculative generation (lossless for greedy).

        Drafts come from the already-seen text via NGramDrafter, verified in
        ONE batched forward pass. On mismatch, truncates KV caches to the
        first divergence and continues. Falls back to single-token decoding
        when no draft exists (or under out-of-core KV mode).

        NOTE on sampling: greedy (temperature=0) is exactly lossless vs
        `generate()`. For temperature>0 the same-sample fast path is used:
        statistically consistent, not distribution-exact rejection sampling.

        Returns (generated_tokens, SpeculativeStats).
        """
        drafter = NGramDrafter(n=n, k=k)
        stats = SpeculativeStats()
        if max_new_tokens <= 0 or len(prompt_tokens) == 0:
            return [], stats

        start_len = 0 if reset_cache else self._sequence_length
        if start_len + len(prompt_tokens) > self.config.max_context_length:
            raise ValueError("Sequence length exceeds max_context_length")
        if start_len + len(prompt_tokens) + max_new_tokens - 1 > self.config.max_context_length:
            raise ValueError("Total generation sequence length exceeds max_context_length")

        if reset_cache:
            self.reset()

        eos_set = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                eos_set = set(int(t) for t in eos_token_id)
            else:
                eos_set = {int(eos_token_id)}

        def _stops(tokens: List[int]) -> Optional[List[int]]:
            for i, t in enumerate(tokens):
                if t in eos_set:
                    return tokens[: i + 1]
            return None

        final_logits = self.prefill(prompt_tokens)
        prev_logits = final_logits
        first_token = self.sample(final_logits, temperature=temperature, top_p=top_p)
        generated: List[int] = [first_token]
        self._last_token = first_token
        self._last_logits = final_logits
        if callback is not None:
            callback(first_token)
        if first_token in eos_set:
            stats.tokens_generated = 1
            return generated, stats

        use_spec = not self._spec_uses_out_of_core()

        # Loop invariant (same as generate()): generated[-1] is sampled but
        # NOT yet fed into the KV cache; cache holds prompt + generated[:-1].
        while len(generated) < max_new_tokens:
            room = max_new_tokens - len(generated)
            context = list(prompt_tokens) + generated
            draft = drafter.propose(context)[:k] if (use_spec and len(context) >= n) else []

            if not draft or room <= 1:
                stats.fallback_steps += 1
                next_tok = self.decode_step(generated[-1], temperature=temperature, top_p=top_p)
                stats.spec_model_calls += 1
                generated.append(next_tok)
                if callback is not None:
                    callback(next_tok)
                if next_tok in eos_set:
                    break
                continue

            # Clip drafts so emitted tokens (drafts + 1 extra) fit in room.
            draft_eff = draft[: room - 1]
            stats.drafts_proposed += 1
            stats.draft_tokens_proposed += len(draft_eff)
            snap = self._spec_caches_snapshot()
            base = snap["seq_len"] if snap else self._sequence_length

            # ONE batched forward: [last_unfed] + drafts. Position i verifies
            # draft[i] (conditioned on full context); last position samples
            # the extra token. Cache ends with base + 1 + D tokens fed.
            batch = [generated[-1]] + draft_eff
            batch_arr = mx.array([batch], dtype=mx.int32)
            logits_batch = self.model(batch_arr, use_cache=True)
            mx.eval(logits_batch)
            m4_bridge_synchronize()
            stats.spec_model_calls += 1

            candidates = [logits_batch[0, i, :] for i in range(len(batch))]
            accepted = 0
            mismatch_tok: Optional[int] = None
            for i, d_tok in enumerate(draft_eff):
                t = self.sample(candidates[i], temperature=temperature, top_p=top_p)
                if t == d_tok:
                    accepted += 1
                else:
                    mismatch_tok = t
                    break

            if accepted == len(draft_eff):
                extra = self.sample(candidates[-1], temperature=temperature, top_p=top_p)
                new_tokens = draft_eff + [extra]
                stats.tokens_accepted += len(draft_eff)
                self._last_logits = candidates[-1]
            else:
                assert mismatch_tok is not None
                # Drop rejected KV tail: keep last_unfed + accepted drafts fed.
                self._spec_caches_truncate(base + 1 + accepted)
                new_tokens = draft_eff[:accepted] + [mismatch_tok]
                stats.tokens_accepted += accepted
                self._last_logits = candidates[accepted]

            stopped = _stops(new_tokens)
            emit = stopped if stopped is not None else new_tokens
            if stopped is not None:
                # Positional accounting: emit is a prefix of new_tokens whose
                # first `accepted` entries (full-accept: all of draft_eff)
                # are fed drafts and whose last entry may be the unfed final.
                n_draft_slots = accepted if mismatch_tok is not None else len(draft_eff)
                eos_idx = len(emit) - 1
                eos_was_fed = eos_idx < n_draft_slots
                n_emitted_drafts = min(len(emit), n_draft_slots)
                fed_target = base + 1 + n_emitted_drafts - (1 if eos_was_fed else 0)
                self._spec_caches_truncate(max(base, fed_target))
            for t in emit:
                generated.append(t)
                if callback is not None:
                    callback(t)
            self._last_token = generated[-1]
            if stopped is not None:
                break

        stats.tokens_generated = len(generated)
        return generated, stats

    def get_memory_footprint_mb(self) -> float:
        """Returns accurate UMA physical memory footprint via MetalUMABridge."""
        return MetalUMABridge.get_instance().get_uma_footprint_mb()
