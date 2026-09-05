"""
BonsaiTokenizer: High-performance BPE tokenizer extracted directly from GGUF metadata.
Backed by tokenizers.Tokenizer with exact BPE vocabulary, merges, and special tokens.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel


class BonsaiTokenizer:
    def __init__(self, gguf_path: str | Path) -> None:
        self.gguf_path = Path(gguf_path)
        self.tokenizer_json_path = self.gguf_path.parent / "tokenizer.json"
        self.bos_token_id = 248044
        self.eos_token_id = 248046
        self.pad_token_id = 248044
        self.vocab_size = 248320
        self._tokenizer: Tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> Tokenizer:
        if self.tokenizer_json_path.exists():
            return Tokenizer.from_file(str(self.tokenizer_json_path))

        with open(self.gguf_path, "rb") as f:
            buf = f.read(32 * 1024 * 1024)

        off = 24
        n_kv = struct.unpack_from("<Q", buf, 16)[0]

        def _str(o: int):
            n = struct.unpack_from("<Q", buf, o)[0]
            o += 8
            return buf[o:o + n].decode("utf-8", "replace"), o + n

        def _skip(o: int, vt: int):
            if vt in (0, 1, 7): return o + 1
            if vt in (2, 3): return o + 2
            if vt in (4, 5, 6): return o + 4
            if vt in (10, 11, 12): return o + 8
            if vt == 8:
                sn = struct.unpack_from("<Q", buf, o)[0]
                return o + 8 + sn
            if vt == 9:
                it = struct.unpack_from("<I", buf, o)[0]
                o += 4
                c = struct.unpack_from("<Q", buf, o)[0]
                o += 8
                for _ in range(c):
                    o = _skip(o, it)
                return o
            raise ValueError(f"Unknown {vt}")

        tokens: List[str] = []
        merges: List[str] = []

        for _ in range(n_kv):
            k, off = _str(off)
            vt = struct.unpack_from("<I", buf, off)[0]
            off += 4
            if k == "tokenizer.ggml.tokens":
                it = struct.unpack_from("<I", buf, off)[0]
                off += 4
                c = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                for _ in range(c):
                    sn = struct.unpack_from("<Q", buf, off)[0]
                    off += 8
                    s = buf[off:off + sn].decode("utf-8", "replace")
                    off += sn
                    tokens.append(s)
            elif k == "tokenizer.ggml.merges":
                it = struct.unpack_from("<I", buf, off)[0]
                off += 4
                c = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                for _ in range(c):
                    sn = struct.unpack_from("<Q", buf, off)[0]
                    off += 8
                    s = buf[off:off + sn].decode("utf-8", "replace")
                    off += sn
                    merges.append(s)
            elif k == "tokenizer.ggml.bos_token_id":
                self.bos_token_id = struct.unpack_from("<I", buf, off)[0]
                off += 4
            elif k == "tokenizer.ggml.eos_token_id":
                self.eos_token_id = struct.unpack_from("<I", buf, off)[0]
                off += 4
            elif k == "tokenizer.ggml.padding_token_id":
                self.pad_token_id = struct.unpack_from("<I", buf, off)[0]
                off += 4
            else:
                off = _skip(off, vt)

        vocab = {t: i for i, t in enumerate(tokens)}
        merge_pairs = [tuple(m.split(" ")) for m in merges if len(m.split(" ")) == 2]

        tok = Tokenizer(BPE(vocab, merge_pairs, unk_token=None))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)
        tok.decoder = ByteLevelDecoder()

        special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]
        tok.add_special_tokens([AddedToken(s, special=True) for s in special])

        try:
            tok.save(str(self.tokenizer_json_path))
        except Exception:
            pass

        return tok

    def encode(self, text: str) -> List[int]:
        enc = self._tokenizer.encode(text)
        return list(enc.ids)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: List[dict], enable_thinking: bool = True) -> str:
        res = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            res.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if enable_thinking:
            res.append("<|im_start|>assistant\n<think>\n")
        else:
            res.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(res)
