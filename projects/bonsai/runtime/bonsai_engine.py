"""
BonsaiExecutionEngine: Complete end-to-end inference orchestrator for Bonsai 27B.
Connects:
- BonsaiWeightStreamer (out-of-core mmap weight streamer)
- 48 Gated-Delta-Net linear attention layers (recurrent delta rule + depthwise conv1d + gated rmsnorm)
- 16 Full GQA layers (QK-norm, Gated-Q, grouped-query attention, paged KV buffer)
- NGramDrafter (lossless prompt-lookup speculative decoding with neural verification)
- BonsaiTokenizer (exact BPE vocabulary, merges, and special tokens)
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_DIR.parent.parent.parent
for p in (RUNTIME_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from bonsai_weight_streamer import BonsaiWeightStreamer
    from bonsai_tokenizer import BonsaiTokenizer
except ImportError:
    from models.bonsai_27b_ternary.runtime.bonsai_weight_streamer import BonsaiWeightStreamer
    from models.bonsai_27b_ternary.runtime.bonsai_tokenizer import BonsaiTokenizer
from src.engine.ngram_drafter import NGramDrafter, SpeculativeStats


class BonsaiExecutionEngine:
    def __init__(
        self,
        gguf_path: str | Path,
        resident_layer_count: int = 2,
        context_length: int = 262144,
    ) -> None:
        self.gguf_path = Path(gguf_path)
        self.tokenizer = BonsaiTokenizer(self.gguf_path)
        self.streamer = BonsaiWeightStreamer(self.gguf_path, resident_layer_count=resident_layer_count)
        self.num_layers = self.streamer.num_layers
        self.context_length = context_length
        self._sequence_length = 0

        # Load accelerated ternary GEMV/GEMM runtime
        dylib_path = Path(__file__).parent / "libdequant.dylib"
        if not dylib_path.exists():
            raise FileNotFoundError(f"Required library {dylib_path} not found. Compile via clang first.")
        self.lib = ctypes.CDLL(str(dylib_path))
        self.lib.gemv_prism_q2_0.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32
        ]
        self.lib.gemv_prism_q2_0.restype = None

        if hasattr(self.lib, "gemm_prism_q2_0"):
            self.lib.gemm_prism_q2_0.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
            ]
            self.lib.gemm_prism_q2_0.restype = None

        # Preallocate recurrent and conv states for 48 linear-attention layers
        self.recurrent_states: List[np.ndarray] = [
            np.zeros((48, 128, 128), dtype=np.float32) for _ in range(64)
        ]
        self.conv_states: List[np.ndarray] = [
            np.zeros((3, 10240), dtype=np.float32) for _ in range(64)
        ]

        # Preallocate paged KV cache for the 16 GQA layers
        self.max_cached_tokens = min(self.context_length, 8192)
        self.kv_k: Dict[int, np.ndarray] = {
            l: np.zeros((self.max_cached_tokens, 4, 256), dtype=np.float16)
            for l in range(3, 64, 4)
        }
        self.kv_v: Dict[int, np.ndarray] = {
            l: np.zeros((self.max_cached_tokens, 4, 256), dtype=np.float16)
            for l in range(3, 64, 4)
        }
        self.kv_len = 0

    def _reset_state(self) -> None:
        """Resets recurrent state, conv state, and KV caches for a new session."""
        for s in self.recurrent_states:
            s.fill(0.0)
        for c in self.conv_states:
            c.fill(0.0)
        for k in self.kv_k.values():
            k.fill(0.0)
        for v in self.kv_v.values():
            v.fill(0.0)
        self.kv_len = 0
        self._sequence_length = 0

    def apply_rope(self, vec: np.ndarray, pos: int) -> np.ndarray:
        rope_dim = 64
        half_dim = 32
        theta = 1.0 / (10000000.0 ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        angles = float(pos) * theta
        cos = np.cos(angles)
        sin = np.sin(angles)
        out = vec.copy()
        q_rot = vec[:, :rope_dim]
        q0 = q_rot[:, :half_dim]
        q1 = q_rot[:, half_dim:]
        out[:, :half_dim] = q0 * cos - q1 * sin
        out[:, half_dim:rope_dim] = q1 * cos + q0 * sin
        return out

    def _get_raw_ptr(self, mv: memoryview) -> ctypes.c_void_p:
        arr = np.frombuffer(mv, dtype=np.uint8)
        return arr.ctypes.data_as(ctypes.c_void_p)

    def embed_token(self, tid: int) -> np.ndarray:
        """Extracts and dequantizes embedding vector for a single token ID."""
        emb = self.streamer.tensors["token_embd.weight"]
        offset = emb.offset + (tid % 248320) * 1360
        mv = memoryview(self.streamer._mm)[offset : offset + 1360]
        blocks = np.frombuffer(mv, dtype=np.uint8).reshape(40, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        qs = blocks[:, 2:]
        w0 = ((qs & 0x03).astype(np.int8) - 1).astype(np.float32)
        w1 = (((qs >> 2) & 0x03).astype(np.int8) - 1).astype(np.float32)
        w2 = (((qs >> 4) & 0x03).astype(np.int8) - 1).astype(np.float32)
        w3 = (((qs >> 6) & 0x03).astype(np.int8) - 1).astype(np.float32)
        unpacked = np.empty((40, 32, 4), dtype=np.float32)
        unpacked[:, :, 0] = w0
        unpacked[:, :, 1] = w1
        unpacked[:, :, 2] = w2
        unpacked[:, :, 3] = w3
        return (unpacked.reshape(40, 128) * scales).reshape(5120).astype(np.float16)

    def prefill(self, prompt_tokens: List[int]) -> np.ndarray:
        """
        Executes layer-by-layer prompt prefill across all prompt tokens.
        Loads each layer once, keeping resident layers <= resident_layer_count.
        """
        self._reset_state()
        P = len(prompt_tokens)
        H = np.stack([self.embed_token(t) for t in prompt_tokens])

        for lyr in range(self.num_layers):
            weights = self.streamer.acquire_layer(lyr)
            is_gqa = ((lyr + 1) % 4 == 0)
            norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_norm.weight"], dtype=np.float32)
            post_w = np.frombuffer(weights[f"blk.{lyr}.post_attention_norm.weight"], dtype=np.float32)

            for t in range(P):
                x = H[t]
                rms = np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-6)
                x_norm = (x.astype(np.float32) / rms * norm_w).astype(np.float16)
                x_norm_ptr = x_norm.ctypes.data_as(ctypes.c_void_p)

                if not is_gqa:
                    u = np.empty(10240, dtype=np.float16)
                    z = np.empty(6144, dtype=np.float16)
                    a = np.empty(48, dtype=np.float16)
                    b = np.empty(48, dtype=np.float16)

                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_qkv.weight"]), u.ctypes.data_as(ctypes.c_void_p), 5120, 10240)
                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_gate.weight"]), z.ctypes.data_as(ctypes.c_void_p), 5120, 6144)
                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ssm_alpha.weight"]), a.ctypes.data_as(ctypes.c_void_p), 5120, 48)
                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ssm_beta.weight"]), b.ctypes.data_as(ctypes.c_void_p), 5120, 48)

                    # Conv1d layout: (10240, 4).T -> (4, 10240)
                    conv_w = np.frombuffer(weights[f"blk.{lyr}.ssm_conv1d.weight"], dtype=np.float32).reshape(10240, 4).T
                    conv_in = np.vstack([self.conv_states[lyr], u.astype(np.float32)])
                    conv_out = np.sum(conv_in * conv_w, axis=0)
                    conv_act = conv_out * (1.0 / (1.0 + np.exp(-np.clip(conv_out, -30.0, 30.0))))
                    self.conv_states[lyr] = conv_in[1:]

                    Q = conv_act[:2048].reshape(16, 128)
                    K = conv_act[2048:4096].reshape(16, 128)
                    V = conv_act[4096:].reshape(48, 128)
                    Q_norm = Q / np.sqrt(np.sum(Q ** 2, axis=-1, keepdims=True) + 1e-6)
                    K_norm = K / np.sqrt(np.sum(K ** 2, axis=-1, keepdims=True) + 1e-6)

                    ssm_a = np.frombuffer(weights[f"blk.{lyr}.ssm_a"], dtype=np.float32)
                    ssm_dt = np.frombuffer(weights[f"blk.{lyr}.ssm_dt.bias"], dtype=np.float32)
                    sp = np.log1p(np.exp(np.clip(a.astype(np.float32) + ssm_dt, -30.0, 30.0)))
                    g = -np.exp(ssm_a) * sp
                    decay = np.exp(np.clip(g, -30.0, 0.0))
                    beta = 1.0 / (1.0 + np.exp(-np.clip(b.astype(np.float32), -30.0, 30.0)))

                    o = np.empty((48, 128), dtype=np.float32)
                    scale = 1.0 / np.sqrt(128.0)
                    S = self.recurrent_states[lyr]
                    for h in range(48):
                        qk = h // 3
                        q_h = Q_norm[qk]
                        k_h = K_norm[qk]
                        v_h = V[h]
                        S[h] *= decay[h]
                        kv_mem = S[h].T @ k_h
                        delta = (v_h - kv_mem) * beta[h]
                        S[h] += np.outer(k_h, delta)
                        o[h] = (S[h].T @ q_h) * scale

                    ssm_norm_w = np.frombuffer(weights[f"blk.{lyr}.ssm_norm.weight"], dtype=np.float32)
                    z_f32 = z.astype(np.float32).reshape(48, 128)
                    silu_z = z_f32 / (1.0 + np.exp(-np.clip(z_f32, -30.0, 30.0)))
                    rms_o = np.sqrt(np.mean(o ** 2, axis=-1, keepdims=True) + 1e-6)
                    o_gated = (o / rms_o * ssm_norm_w * silu_z).reshape(6144).astype(np.float16)

                    attn_out = np.empty(5120, dtype=np.float16)
                    self.lib.gemv_prism_q2_0(o_gated.ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.ssm_out.weight"]), attn_out.ctypes.data_as(ctypes.c_void_p), 6144, 5120)
                    x1 = (x.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)

                else:
                    q_raw = np.empty(12288, dtype=np.float16)
                    k_raw = np.empty(1024, dtype=np.float16)
                    v_raw = np.empty(1024, dtype=np.float16)

                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_q.weight"]), q_raw.ctypes.data_as(ctypes.c_void_p), 5120, 12288)
                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_k.weight"]), k_raw.ctypes.data_as(ctypes.c_void_p), 5120, 1024)
                    self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_v.weight"]), v_raw.ctypes.data_as(ctypes.c_void_p), 5120, 1024)

                    q_f32 = q_raw.astype(np.float32).reshape(24, 512)
                    query_states = q_f32[:, :256]
                    gate_states = q_f32[:, 256:]

                    q_norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_q_norm.weight"], dtype=np.float32)
                    k_norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_k_norm.weight"], dtype=np.float32)
                    q_normed = query_states / np.sqrt(np.mean(query_states ** 2, axis=-1, keepdims=True) + 1e-6) * q_norm_w

                    k_f32 = k_raw.astype(np.float32).reshape(4, 256)
                    k_normed = k_f32 / np.sqrt(np.mean(k_f32 ** 2, axis=-1, keepdims=True) + 1e-6) * k_norm_w

                    q_rope = self.apply_rope(q_normed, t)
                    k_rope = self.apply_rope(k_normed, t)

                    self.kv_k[lyr][t] = k_rope.astype(np.float16)
                    self.kv_v[lyr][t] = v_raw.reshape(4, 256)

                    attn_out = np.empty((24, 256), dtype=np.float32)
                    scale = 1.0 / 16.0
                    k_hist = self.kv_k[lyr][:t + 1].astype(np.float32)
                    v_hist = self.kv_v[lyr][:t + 1].astype(np.float32)

                    for h in range(24):
                        kv_h = h // 6
                        scores = (k_hist[:, kv_h, :] @ q_rope[h]) * scale
                        scores_exp = np.exp(scores - np.max(scores))
                        w_attn = scores_exp / np.sum(scores_exp)
                        attn_out[h] = np.sum(w_attn[:, None] * v_hist[:, kv_h, :], axis=0)

                    attn_out = attn_out * (1.0 / (1.0 + np.exp(-np.clip(gate_states, -30.0, 30.0))))

                    attn_proj = np.empty(5120, dtype=np.float16)
                    self.lib.gemv_prism_q2_0(attn_out.reshape(6144).astype(np.float16).ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.attn_output.weight"]), attn_proj.ctypes.data_as(ctypes.c_void_p), 6144, 5120)
                    x1 = (x.astype(np.float32) + attn_proj.astype(np.float32)).astype(np.float16)

                # MLP
                rms_p = np.sqrt(np.mean(x1.astype(np.float32) ** 2) + 1e-6)
                x1_norm = (x1.astype(np.float32) / rms_p * post_w).astype(np.float16)
                x1_norm_ptr = x1_norm.ctypes.data_as(ctypes.c_void_p)

                f_gate = np.empty(17408, dtype=np.float16)
                f_up = np.empty(17408, dtype=np.float16)
                self.lib.gemv_prism_q2_0(x1_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ffn_gate.weight"]), f_gate.ctypes.data_as(ctypes.c_void_p), 5120, 17408)
                self.lib.gemv_prism_q2_0(x1_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ffn_up.weight"]), f_up.ctypes.data_as(ctypes.c_void_p), 5120, 17408)

                g_f32 = f_gate.astype(np.float32)
                act = (g_f32 / (1.0 + np.exp(-np.clip(g_f32, -30.0, 30.0))) * f_up.astype(np.float32)).astype(np.float16)
                mlp_out = np.empty(5120, dtype=np.float16)
                self.lib.gemv_prism_q2_0(act.ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.ffn_down.weight"]), mlp_out.ctypes.data_as(ctypes.c_void_p), 17408, 5120)

                H[t] = (x1.astype(np.float32) + mlp_out.astype(np.float32)).astype(np.float16)

        self.kv_len = P
        self._sequence_length = P

        x_last = H[-1]
        out_norm_entry = self.streamer.tensors["output_norm.weight"]
        out_norm_w = np.frombuffer(memoryview(self.streamer._mm)[out_norm_entry.offset : out_norm_entry.offset + out_norm_entry.nbytes], dtype=np.float32)
        rms_out = np.sqrt(np.mean(x_last.astype(np.float32) ** 2) + 1e-6)
        x_final = (x_last.astype(np.float32) / rms_out * out_norm_w).astype(np.float16)

        out_entry = self.streamer.tensors["output.weight"]
        logits = np.empty(248320, dtype=np.float16)
        raw_out = np.frombuffer(memoryview(self.streamer._mm)[out_entry.offset : out_entry.offset + out_entry.nbytes], dtype=np.uint8)
        self.lib.gemv_prism_q2_0(x_final.ctypes.data_as(ctypes.c_void_p), raw_out.ctypes.data_as(ctypes.c_void_p), logits.ctypes.data_as(ctypes.c_void_p), 5120, 248320)
        return logits.astype(np.float32)

    def forward_step(self, token_id: int) -> np.ndarray:
        """
        Executes a full 64-layer forward pass for a single token (M = 1).
        Updates recurrent and KV states in-place and returns vocabulary logits [248320].
        """
        x = self.embed_token(token_id)
        current_kv_pos = min(self.kv_len, self.max_cached_tokens - 1)

        for lyr in range(self.num_layers):
            weights = self.streamer.acquire_layer(lyr)
            is_gqa = ((lyr + 1) % 4 == 0)

            # Pre-attention RMSNorm
            norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_norm.weight"], dtype=np.float32)
            rms = np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-6)
            x_norm = (x.astype(np.float32) / rms * norm_w).astype(np.float16)
            x_norm_ptr = x_norm.ctypes.data_as(ctypes.c_void_p)

            if not is_gqa:
                # 48 Gated-Delta-Net Linear Attention Layers
                u = np.empty(10240, dtype=np.float16)
                z = np.empty(6144, dtype=np.float16)
                a = np.empty(48, dtype=np.float16)
                b = np.empty(48, dtype=np.float16)

                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_qkv.weight"]), u.ctypes.data_as(ctypes.c_void_p), 5120, 10240)
                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_gate.weight"]), z.ctypes.data_as(ctypes.c_void_p), 5120, 6144)
                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ssm_alpha.weight"]), a.ctypes.data_as(ctypes.c_void_p), 5120, 48)
                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ssm_beta.weight"]), b.ctypes.data_as(ctypes.c_void_p), 5120, 48)

                # Depthwise Causal Conv1D (kernel=4) + SiLU: (10240, 4).T -> (4, 10240)
                conv_w = np.frombuffer(weights[f"blk.{lyr}.ssm_conv1d.weight"], dtype=np.float32).reshape(10240, 4).T
                conv_in = np.vstack([self.conv_states[lyr], u.astype(np.float32)])
                conv_out = np.sum(conv_in * conv_w, axis=0)
                conv_act = conv_out * (1.0 / (1.0 + np.exp(-np.clip(conv_out, -30.0, 30.0))))
                self.conv_states[lyr] = conv_in[1:]

                # Q/K L2-Normalization per head
                Q = conv_act[:2048].reshape(16, 128)
                K = conv_act[2048:4096].reshape(16, 128)
                V = conv_act[4096:].reshape(48, 128)
                Q_norm = Q / np.sqrt(np.sum(Q ** 2, axis=-1, keepdims=True) + 1e-6)
                K_norm = K / np.sqrt(np.sum(K ** 2, axis=-1, keepdims=True) + 1e-6)

                # Decay gate g_t computation
                ssm_a = np.frombuffer(weights[f"blk.{lyr}.ssm_a"], dtype=np.float32)
                ssm_dt = np.frombuffer(weights[f"blk.{lyr}.ssm_dt.bias"], dtype=np.float32)
                sp = np.log1p(np.exp(np.clip(a.astype(np.float32) + ssm_dt, -30.0, 30.0)))
                g = -np.exp(ssm_a) * sp
                decay = np.exp(np.clip(g, -30.0, 0.0))
                beta = 1.0 / (1.0 + np.exp(-np.clip(b.astype(np.float32), -30.0, 30.0)))

                # Recurrent Delta Rule Core
                o = np.empty((48, 128), dtype=np.float32)
                scale = 1.0 / np.sqrt(128.0)
                S = self.recurrent_states[lyr]
                for h in range(48):
                    qk = h // 3
                    q_h = Q_norm[qk]
                    k_h = K_norm[qk]
                    v_h = V[h]
                    S[h] *= decay[h]
                    kv_mem = S[h].T @ k_h
                    delta = (v_h - kv_mem) * beta[h]
                    S[h] += np.outer(k_h, delta)
                    o[h] = (S[h].T @ q_h) * scale

                # Gated RMSNorm: RMSNorm(O) * SiLU(Z)
                ssm_norm_w = np.frombuffer(weights[f"blk.{lyr}.ssm_norm.weight"], dtype=np.float32)
                z_f32 = z.astype(np.float32).reshape(48, 128)
                silu_z = z_f32 / (1.0 + np.exp(-np.clip(z_f32, -30.0, 30.0)))
                rms_o = np.sqrt(np.mean(o ** 2, axis=-1, keepdims=True) + 1e-6)
                o_gated = (o / rms_o * ssm_norm_w * silu_z).reshape(6144).astype(np.float16)

                # SSM out projection & residual
                attn_out = np.empty(5120, dtype=np.float16)
                self.lib.gemv_prism_q2_0(o_gated.ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.ssm_out.weight"]), attn_out.ctypes.data_as(ctypes.c_void_p), 6144, 5120)
                x1 = (x.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)

            else:
                # 16 Full GQA Layers
                q_raw = np.empty(12288, dtype=np.float16)
                k_raw = np.empty(1024, dtype=np.float16)
                v_raw = np.empty(1024, dtype=np.float16)

                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_q.weight"]), q_raw.ctypes.data_as(ctypes.c_void_p), 5120, 12288)
                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_k.weight"]), k_raw.ctypes.data_as(ctypes.c_void_p), 5120, 1024)
                self.lib.gemv_prism_q2_0(x_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.attn_v.weight"]), v_raw.ctypes.data_as(ctypes.c_void_p), 5120, 1024)

                q_f32 = q_raw.astype(np.float32).reshape(24, 512)
                query_states = q_f32[:, :256]
                gate_states = q_f32[:, 256:]

                # QK-Norm per head
                q_norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_q_norm.weight"], dtype=np.float32)
                k_norm_w = np.frombuffer(weights[f"blk.{lyr}.attn_k_norm.weight"], dtype=np.float32)
                q_normed = query_states / np.sqrt(np.mean(query_states ** 2, axis=-1, keepdims=True) + 1e-6) * q_norm_w

                k_f32 = k_raw.astype(np.float32).reshape(4, 256)
                k_normed = k_f32 / np.sqrt(np.mean(k_f32 ** 2, axis=-1, keepdims=True) + 1e-6) * k_norm_w

                q_rope = self.apply_rope(q_normed, current_kv_pos)
                k_rope = self.apply_rope(k_normed, current_kv_pos)

                # Cache K and V
                self.kv_k[lyr][current_kv_pos] = k_rope.astype(np.float16)
                self.kv_v[lyr][current_kv_pos] = v_raw.reshape(4, 256)

                # Grouped-Query Attention
                attn_out = np.empty((24, 256), dtype=np.float32)
                scale = 1.0 / 16.0
                k_hist = self.kv_k[lyr][:current_kv_pos + 1].astype(np.float32)
                v_hist = self.kv_v[lyr][:current_kv_pos + 1].astype(np.float32)

                for h in range(24):
                    kv_h = h // 6
                    scores = (k_hist[:, kv_h, :] @ q_rope[h]) * scale
                    scores_exp = np.exp(scores - np.max(scores))
                    w_attn = scores_exp / np.sum(scores_exp)
                    attn_out[h] = np.sum(w_attn[:, None] * v_hist[:, kv_h, :], axis=0)

                # Gate applied AFTER attention using sigmoid: attn_out * sigmoid(gate_states)
                attn_out = attn_out * (1.0 / (1.0 + np.exp(-np.clip(gate_states, -30.0, 30.0))))

                # GQA Out Projection & residual
                attn_proj = np.empty(5120, dtype=np.float16)
                self.lib.gemv_prism_q2_0(attn_out.reshape(6144).astype(np.float16).ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.attn_output.weight"]), attn_proj.ctypes.data_as(ctypes.c_void_p), 6144, 5120)
                x1 = (x.astype(np.float32) + attn_proj.astype(np.float32)).astype(np.float16)

            # Post-Attention RMSNorm + SwiGLU MLP for all 64 layers
            post_w = np.frombuffer(weights[f"blk.{lyr}.post_attention_norm.weight"], dtype=np.float32)
            rms_p = np.sqrt(np.mean(x1.astype(np.float32) ** 2) + 1e-6)
            x1_norm = (x1.astype(np.float32) / rms_p * post_w).astype(np.float16)
            x1_norm_ptr = x1_norm.ctypes.data_as(ctypes.c_void_p)

            f_gate = np.empty(17408, dtype=np.float16)
            f_up = np.empty(17408, dtype=np.float16)
            self.lib.gemv_prism_q2_0(x1_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ffn_gate.weight"]), f_gate.ctypes.data_as(ctypes.c_void_p), 5120, 17408)
            self.lib.gemv_prism_q2_0(x1_norm_ptr, self._get_raw_ptr(weights[f"blk.{lyr}.ffn_up.weight"]), f_up.ctypes.data_as(ctypes.c_void_p), 5120, 17408)

            g_f32 = f_gate.astype(np.float32)
            act = (g_f32 / (1.0 + np.exp(-np.clip(g_f32, -30.0, 30.0))) * f_up.astype(np.float32)).astype(np.float16)
            mlp_out = np.empty(5120, dtype=np.float16)
            self.lib.gemv_prism_q2_0(act.ctypes.data_as(ctypes.c_void_p), self._get_raw_ptr(weights[f"blk.{lyr}.ffn_down.weight"]), mlp_out.ctypes.data_as(ctypes.c_void_p), 17408, 5120)

            x = (x1.astype(np.float32) + mlp_out.astype(np.float32)).astype(np.float16)

        self.kv_len += 1
        self._sequence_length += 1

        # Final RMSNorm
        out_norm_entry = self.streamer.tensors["output_norm.weight"]
        out_norm_w = np.frombuffer(memoryview(self.streamer._mm)[out_norm_entry.offset : out_norm_entry.offset + out_norm_entry.nbytes], dtype=np.float32)
        rms_out = np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-6)
        x_final = (x.astype(np.float32) / rms_out * out_norm_w).astype(np.float16)

        # LM Head Projection
        out_entry = self.streamer.tensors["output.weight"]
        logits = np.empty(248320, dtype=np.float16)
        raw_out = np.frombuffer(memoryview(self.streamer._mm)[out_entry.offset : out_entry.offset + out_entry.nbytes], dtype=np.uint8)
        self.lib.gemv_prism_q2_0(x_final.ctypes.data_as(ctypes.c_void_p), raw_out.ctypes.data_as(ctypes.c_void_p), logits.ctypes.data_as(ctypes.c_void_p), 5120, 248320)
        return logits.astype(np.float32)

    def sample(self, logits: np.ndarray, temperature: float = 0.7, top_k: int = 20, top_p: float = 0.95) -> int:
        """Samples token from unnormalized logits with optional temperature, top-k, and top-p."""
        if temperature <= 1e-6:
            return int(np.argmax(logits))

        scaled = logits / max(float(temperature), 1e-5)
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            scaled[indices_to_remove] = -np.inf

        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs /= sum_p
        else:
            return int(np.argmax(logits))

        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_p = probs[sorted_idx]
            cum_p = np.cumsum(sorted_p)
            cutoff = cum_p > top_p
            cutoff[0] = False
            sorted_p[cutoff] = 0.0
            sum_sp = np.sum(sorted_p)
            if sum_sp > 0:
                sorted_p /= sum_sp
                return int(sorted_idx[np.random.choice(len(sorted_p), p=sorted_p)])

        return int(np.random.choice(len(probs), p=probs))

    def generate_stream(
        self,
        prompt: Union[str, List[dict]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.95,
        n: int = 3,
        k: int = 4,
        use_speculative: bool = True,
    ):
        """
        Yields text chunks incrementally as tokens are generated and verified.
        """
        if isinstance(prompt, list):
            prompt_text = self.tokenizer.apply_chat_template(prompt)
        else:
            prompt_text = prompt

        prompt_tokens = self.tokenizer.encode(prompt_text)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_token_id]

        drafter = NGramDrafter(n=n, k=k)

        # Prefill: stream prompt tokens through the model layer-by-layer to prime states
        logits = self.prefill(prompt_tokens)

        stop_token_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 248044, 248046}
        generated_tokens: List[int] = []
        curr_context = list(prompt_tokens)
        prev_text = ""

        while len(generated_tokens) < max_tokens:
            room = max_tokens - len(generated_tokens)
            draft = drafter.propose(curr_context)[:k] if (use_speculative and len(curr_context) >= n) else []

            if draft and room > 1:
                all_accepted = True
                for candidate in draft[:room]:
                    model_tok = self.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    if candidate == model_tok:
                        generated_tokens.append(candidate)
                        curr_context.append(candidate)

                        cur_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        delta = cur_text[len(prev_text):]
                        prev_text = cur_text
                        if delta:
                            yield delta

                        if candidate in stop_token_ids:
                            all_accepted = False
                            break
                        logits = self.forward_step(candidate)
                    else:
                        generated_tokens.append(model_tok)
                        curr_context.append(model_tok)
                        all_accepted = False

                        cur_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        delta = cur_text[len(prev_text):]
                        prev_text = cur_text
                        if delta:
                            yield delta

                        if model_tok in stop_token_ids:
                            break
                        logits = self.forward_step(model_tok)
                        break

                if not all_accepted and (generated_tokens and generated_tokens[-1] in stop_token_ids):
                    break
            else:
                next_t = self.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                generated_tokens.append(next_t)
                curr_context.append(next_t)

                cur_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                delta = cur_text[len(prev_text):]
                prev_text = cur_text
                if delta:
                    yield delta

                if next_t in stop_token_ids:
                    break
                if len(generated_tokens) >= max_tokens:
                    break
                logits = self.forward_step(next_t)

    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.95,
        n: int = 3,
        k: int = 4,
        use_speculative: bool = True,
    ) -> Tuple[str, SpeculativeStats]:
        """
        Generates text using out-of-core streaming weights with verified prompt-lookup
        speculative decoding. Real model logits verify all draft candidates.
        """
        chunks = []
        for delta in self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            n=n,
            k=k,
            use_speculative=use_speculative,
        ):
            chunks.append(delta)

        stats = SpeculativeStats(tokens_generated=len(chunks))
        return "".join(chunks), stats
