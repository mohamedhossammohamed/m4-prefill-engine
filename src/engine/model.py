"""
Transformer Model and Block Architecture for Apple Silicon M4 / MLX.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from src.engine.config import EngineConfig
from src.engine.modules import create_kv_cache, create_linear, m4_bridge_synchronize


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization supporting standard LLaMA RMSNorm
    and Gemma-style 'gemma_add_one' normalization.
    """

    def __init__(self, dims: int, eps: float = 1e-5, norm_type: str = "rmsnorm") -> None:
        super().__init__()
        self.dims = int(dims)
        self.eps = float(eps)
        self.norm_type = norm_type.lower().strip()
        if self.norm_type == "gemma_add_one":
            self.weight = mx.zeros((dims,), dtype=mx.float16)
        else:
            self.weight = mx.ones((dims,), dtype=mx.float16)

    def init_synthetic_weights(self) -> None:
        if self.norm_type == "gemma_add_one":
            self.weight = mx.zeros((self.dims,), dtype=mx.float16)
        else:
            self.weight = mx.ones((self.dims,), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        variance = mx.mean(mx.square(x_f32), axis=-1, keepdims=True)
        normed = (x_f32 * mx.rsqrt(variance + self.eps)).astype(x.dtype)
        if self.norm_type == "gemma_add_one":
            return normed * (1.0 + self.weight)
        return normed * self.weight


class Embedding(nn.Module):
    """
    Embedding layer gathering token IDs zero-copy in unified memory.
    Uses CPU stream for index gathering to avoid dynamic JIT metal kernel builds.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.weight = mx.zeros((self.vocab_size, self.hidden_dim), dtype=mx.float16)

    def __call__(self, tokens: mx.array) -> mx.array:
        if bool(mx.any(tokens < 0).item()) or bool(mx.any(tokens >= self.vocab_size).item()):
            raise IndexError(f"Token ID out of vocabulary bounds [0, {self.vocab_size})")
        return mx.take(self.weight, tokens, axis=0, stream=mx.cpu)


class TransformerBlock(nn.Module):
    """
    Transformer block layer with pluggable linear projections and KV cache:
    Pre-attention RMSNorm -> QKV Projections -> RoPE -> GQA Attention & KV Cache
    -> Out Projection -> Residual Add -> Post-attention RMSNorm -> MLP -> Residual Add.
    """

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = config.scale

        # 1. Pre-attention normalization
        self.input_norm = RMSNorm(
            dims=config.hidden_dim,
            eps=config.norm_eps,
            norm_type=config.norm_type,
        )

        # 2. QKV Linear Projections
        self.wq = create_linear(
            in_features=config.hidden_dim,
            out_features=config.q_dim,
            format=config.weight_format,
            backend=config.backend,
        )
        self.wk = create_linear(
            in_features=config.hidden_dim,
            out_features=config.kv_dim,
            format=config.weight_format,
            backend=config.backend,
        )
        self.wv = create_linear(
            in_features=config.hidden_dim,
            out_features=config.kv_dim,
            format=config.weight_format,
            backend=config.backend,
        )
        self.wo = create_linear(
            in_features=config.q_dim,
            out_features=config.hidden_dim,
            format=config.weight_format,
            backend=config.backend,
        )

        # 3. Rotary Position Embeddings (RoPE)
        self.rope = nn.RoPE(
            dims=config.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

        # 4. KV Cache (M4 hardware circular/out-of-core or MLX baseline)
        self.kv_cache = create_kv_cache(
            head_dim=config.head_dim,
            n_heads=config.num_kv_heads,
            backend=config.backend,
            mode=config.kv_mode,
            max_seq_len=config.max_context_length,
        )

        # 5. Post-attention normalization
        self.post_attention_norm = RMSNorm(
            dims=config.hidden_dim,
            eps=config.norm_eps,
            norm_type=config.norm_type,
        )

        # 6. MLP Projections (Gate, Up, Down)
        self.w_gate = create_linear(
            in_features=config.hidden_dim,
            out_features=config.intermediate_dim,
            format=config.weight_format,
            backend=config.backend,
        )
        self.w_up = create_linear(
            in_features=config.hidden_dim,
            out_features=config.intermediate_dim,
            format=config.weight_format,
            backend=config.backend,
        )
        self.w_down = create_linear(
            in_features=config.intermediate_dim,
            out_features=config.hidden_dim,
            format=config.weight_format,
            backend=config.backend,
        )

    def reset(self) -> None:
        """Resets the KV cache for this transformer layer."""
        self.kv_cache.reset()

    def init_synthetic_weights(
        self, scale: Optional[float] = None, seed: Optional[int] = None, unit_weights: bool = False
    ) -> None:
        """Initializes all sub-layer weights deterministically."""
        base_seed = seed if seed is not None else 42
        actual_scale = scale if scale is not None else (0.5 / math.sqrt(self.config.hidden_dim))
        self.input_norm.init_synthetic_weights()
        self.post_attention_norm.init_synthetic_weights()

        for idx, layer in enumerate([
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.w_gate,
            self.w_up,
            self.w_down,
        ]):
            if hasattr(layer, "init_synthetic_weights"):
                layer.init_synthetic_weights(
                    scale=actual_scale, unit_weights=unit_weights, seed=base_seed + idx * 7
                )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Union[str, mx.array]] = None,
        cache: Optional[Any] = None,
        use_cache: bool = True,
    ) -> mx.array:
        """
        Forward pass through the transformer block.
        Accepts 2D [M, K] or 3D [B, M, K] tensors.
        When use_cache=False, computes attention without updating or reading from KV cache.
        """
        is_2d = x.ndim == 2
        if is_2d:
            x = x[None, ...]

        B, M, K = x.shape

        # Pre-attention normalization
        norm_x = self.input_norm(x)

        # QKV Projections
        q_proj = self.wq(norm_x)
        k_proj = self.wk(norm_x)
        v_proj = self.wv(norm_x)
        if self.config.backend == "m4":
            m4_bridge_synchronize()

        # Reshape into head dimensions: [B, num_heads, M, head_dim]
        q = q_proj.reshape(B, M, self.config.num_q_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        k = k_proj.reshape(B, M, self.config.num_kv_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        v = v_proj.reshape(B, M, self.config.num_kv_heads, self.config.head_dim).transpose(0, 2, 1, 3)

        if use_cache:
            active_cache = cache if cache is not None else self.kv_cache
            # Apply RoPE relative to current KV cache offset
            offset = active_cache.offset
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

            # Update KV cache and fetch full context
            k_cached, v_cached = active_cache.update_and_fetch(k, v)
        else:
            # Stateless evaluation: RoPE from offset 0, no cache update
            q = self.rope(q, offset=0)
            k = self.rope(k, offset=0)
            k_cached, v_cached = k, v

        # Scaled Dot-Product Attention with GQA support
        if mask is None:
            mask_to_use = "causal" if M > 1 else None
        else:
            mask_to_use = mask

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k_cached, v_cached, scale=self.scale, mask=mask_to_use
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, M, self.config.q_dim)

        # Output projection and residual connection
        wo_out = self.wo(attn_out)
        if self.config.backend == "m4":
            m4_bridge_synchronize()
        x = x + wo_out

        # Post-attention normalization
        norm_x2 = self.post_attention_norm(x)

        # MLP Block: SwiGLU / GeGLU
        gate = self.w_gate(norm_x2)
        up = self.w_up(norm_x2)
        if self.config.backend == "m4":
            m4_bridge_synchronize()

        if self.config.activation == "swiglu":
            # SiLU(gate) * up = gate * sigmoid(gate) * up
            act = gate * mx.sigmoid(gate.astype(mx.float32)).astype(gate.dtype)
        else:  # "geglu"
            # GELU(gate) * up
            act = 0.5 * gate * (1.0 + mx.erf(gate.astype(mx.float32) / math.sqrt(2.0))).astype(gate.dtype)

        act_up = (act.astype(mx.float32) * up.astype(mx.float32)).astype(gate.dtype)
        mlp_out = self.w_down(act_up)
        if self.config.backend == "m4":
            m4_bridge_synchronize()
        x = x + mlp_out

        if is_2d:
            x = x[0]
        return x


class TransformerModel(nn.Module):
    """
    Complete unified transformer model architecture:
    - Embedding layer
    - Stack of L TransformerBlock layers
    - Final RMSNorm
    - LM Head projection to vocab_size
    """

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = Embedding(config.vocab_size, config.hidden_dim)
        self.layers: List[TransformerBlock] = [
            TransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.norm = RMSNorm(
            dims=config.hidden_dim,
            eps=config.norm_eps,
            norm_type=config.norm_type,
        )
        self.lm_head = create_linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
            format=config.weight_format,
            backend=config.backend,
        )

    def reset(self) -> None:
        """Resets all KV caches across all transformer layers."""
        for layer in self.layers:
            layer.reset()

    def init_synthetic_weights(
        self, scale: Optional[float] = None, seed: Optional[int] = 42, unit_weights: bool = False
    ) -> None:
        """
        Initializes synthetic weights deterministically for testing without
        external weight files.
        """
        base_seed = seed if seed is not None else 42
        actual_scale = scale if scale is not None else (0.5 / math.sqrt(self.config.hidden_dim))
        rng = np.random.RandomState(base_seed)

        # Initialize embedding weights
        emb_np = (rng.randn(self.config.vocab_size, self.config.hidden_dim) * 0.02).astype(np.float16)
        self.embedding.weight = mx.array(emb_np)

        # Initialize each transformer block
        for i, layer in enumerate(self.layers):
            layer_seed = base_seed + (i + 1) * 100
            layer.init_synthetic_weights(scale=actual_scale, seed=layer_seed, unit_weights=unit_weights)

        # Initialize final norm
        self.norm.init_synthetic_weights()

        # Initialize LM head
        if hasattr(self.lm_head, "init_synthetic_weights"):
            self.lm_head.init_synthetic_weights(
                scale=actual_scale, seed=base_seed + 9999, unit_weights=unit_weights
            )

    def __call__(
        self,
        tokens: Union[List[int], mx.array],
        mask: Optional[Union[str, mx.array]] = None,
        use_cache: bool = True,
    ) -> mx.array:
        """
        Runs forward pass given input token IDs and returns unnormalized logits.
        """
        if isinstance(tokens, list):
            tokens_arr = mx.array(tokens, dtype=mx.int32)
        else:
            tokens_arr = tokens.astype(mx.int32)

        is_1d = tokens_arr.ndim == 1
        if is_1d:
            tokens_arr = tokens_arr[None, :]

        # 1. Embedding lookup
        h = self.embedding(tokens_arr).astype(mx.float16)

        # 2. Sequential transformer block execution
        for layer in self.layers:
            h = layer(h, mask=mask, use_cache=use_cache)

        # 3. Final normalization
        h = self.norm(h)

        # 4. LM Head projection
        logits = self.lm_head(h)
        if self.config.backend == "m4":
            m4_bridge_synchronize()
        return logits
