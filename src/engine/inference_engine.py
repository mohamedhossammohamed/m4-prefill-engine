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

    def get_memory_footprint_mb(self) -> float:
        """Returns accurate UMA physical memory footprint via MetalUMABridge."""
        return MetalUMABridge.get_instance().get_uma_footprint_mb()
