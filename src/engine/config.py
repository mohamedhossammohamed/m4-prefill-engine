"""
Engine Configuration for Unified M4 Prefill and Autoregressive Generation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union

from src.engine.modules import QuantFormat, normalize_format


@dataclass
class EngineConfig:
    """
    Configuration specification for transformer models and inference execution.
    Defaults to M4 hardware-accelerated backend with GQA and SwiGLU.
    """
    num_layers: int = 4
    hidden_dim: int = 512
    num_q_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    intermediate_dim: int = 1024
    vocab_size: int = 1024
    rope_theta: float = 10000.0
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    backend: str = "m4"
    kv_mode: str = "in_ram"
    weight_format: Union[int, QuantFormat, str] = 0
    max_context_length: int = 2048
    norm_eps: float = 1e-5
    rope_traditional: bool = False

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_q_heads <= 0:
            raise ValueError(f"num_q_heads must be > 0, got {self.num_q_heads}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {self.num_kv_heads}")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA/MHA"
            )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be > 0, got {self.intermediate_dim}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {self.max_context_length}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

        self.backend = self.backend.lower().strip()
        if self.backend not in ("m4", "mlx"):
            raise ValueError(f"Unsupported backend '{self.backend}'. Must be 'm4' or 'mlx'.")

        self.kv_mode = self.kv_mode.lower().strip()
        if self.kv_mode not in ("in_ram", "out_of_core"):
            raise ValueError(f"Unsupported kv_mode '{self.kv_mode}'. Must be 'in_ram' or 'out_of_core'.")

        self.activation = self.activation.lower().strip()
        if self.activation not in ("swiglu", "geglu"):
            raise ValueError(f"Unsupported activation '{self.activation}'. Must be 'swiglu' or 'geglu'.")

        self.norm_type = self.norm_type.lower().strip()
        if self.norm_type not in ("rmsnorm", "gemma_add_one"):
            raise ValueError(f"Unsupported norm_type '{self.norm_type}'. Must be 'rmsnorm' or 'gemma_add_one'.")

        # Normalize format to QuantFormat
        self.weight_format = normalize_format(self.weight_format)

    @property
    def scale(self) -> float:
        """Attention scaling factor: 1.0 / sqrt(head_dim)."""
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def q_dim(self) -> int:
        """Total query projection dimension."""
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Total key/value projection dimension."""
        return self.num_kv_heads * self.head_dim
