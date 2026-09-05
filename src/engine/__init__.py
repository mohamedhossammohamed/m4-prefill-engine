"""
M4 Prefill Engine - Modular Architecture.
"""

from src.engine.config import EngineConfig
from src.engine.model import TransformerBlock, TransformerModel
from src.engine.inference_engine import InferenceEngine
from src.engine.ngram_drafter import NGramDrafter, SpeculativeStats
from src.engine.modules import (
    FORMAT_SPECS,
    M4KVCache,
    M4QuantizedLinear,
    MLXKVCache,
    MLXQuantizedLinear,
    QuantFormat,
    compute_quantized_weight_bytes,
    create_kv_cache,
    create_linear,
    dequantize_to_fp16_matrix,
    generate_synthetic_quantized_weights,
    normalize_format,
)

__all__ = [
    "QuantFormat",
    "FORMAT_SPECS",
    "normalize_format",
    "compute_quantized_weight_bytes",
    "generate_synthetic_quantized_weights",
    "dequantize_to_fp16_matrix",
    "M4QuantizedLinear",
    "MLXQuantizedLinear",
    "M4KVCache",
    "MLXKVCache",
    "create_linear",
    "create_kv_cache",
    "EngineConfig",
    "TransformerBlock",
    "TransformerModel",
    "InferenceEngine",
    "NGramDrafter",
    "SpeculativeStats",
]

