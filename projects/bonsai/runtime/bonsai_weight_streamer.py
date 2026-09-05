"""
BonsaiWeightStreamer: Model-scoped weight streaming coordinator for Bonsai 27B.
Supports standard 'model.layers.N.' and llama.cpp/Bonsai 'blk.N.' tensor prefixes.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def parse_bonsai_layer_index(name: str) -> int:
    """
    Parses layer index from tensor name supporting:
    - 'model.layers.N.'
    - 'blk.N.'
    Returns -1 for shared tensors (e.g. token_embd, output_norm, etc.)
    """
    prefixes = ["model.layers.", "blk."]
    matched_prefix = None
    for p in prefixes:
        if name.startswith(p):
            matched_prefix = p
            break
    if not matched_prefix:
        return -1

    rest = name[len(matched_prefix):]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits or not rest[len(digits):].startswith("."):
        return -1
    return int(digits)


@dataclass
class BonsaiTensorEntry:
    name: str
    offset: int
    nbytes: int
    dims: List[int]
    ttype: int


class BonsaiWeightStreamer:
    """Zero-copy mmap layer streamer for Bonsai 27B weights."""

    def __init__(self, path: str | Path, resident_layer_count: int = 2) -> None:
        self.path = Path(path)
        self.resident_layer_count = int(resident_layer_count)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors: Dict[str, BonsaiTensorEntry] = {}
        self.groups: Dict[int, List[str]] = {}
        self._resident: OrderedDict[int, None] = OrderedDict()
        self._parse()

        self._shared_bytes = sum(self.tensors[n].nbytes for n in self.groups.get(-1, []))
        self._current_bytes = self._shared_bytes
        self._peak_bytes = self._current_bytes
        self._advise_shared()

    def _parse(self) -> None:
        buf = self._mm
        if bytes(buf[0:4]) != b"GGUF":
            raise ValueError("Bad GGUF magic")
        version = struct.unpack_from("<I", buf, 4)[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        n_tensors = struct.unpack_from("<Q", buf, 8)[0]
        n_kv = struct.unpack_from("<Q", buf, 16)[0]

        alignment = 32
        off = 24

        def _str(o):
            n = struct.unpack_from("<Q", buf, o)[0]
            o += 8
            return buf[o:o + n].decode("utf-8", "replace"), o + n

        def _skip(o, vtype):
            if vtype in (0, 1, 7): return o + 1
            if vtype in (2, 3): return o + 2
            if vtype in (4, 5, 6): return o + 4
            if vtype in (10, 11, 12): return o + 8
            if vtype == 8:
                s, o = _str(o); return o
            if vtype == 9:
                it = struct.unpack_from("<I", buf, o)[0]; o += 4
                c = struct.unpack_from("<Q", buf, o)[0]; o += 8
                for _ in range(c): o = _skip(o, it)
                return o
            raise ValueError(f"Unknown vtype {vtype}")

        for _ in range(n_kv):
            k, off = _str(off)
            vt = struct.unpack_from("<I", buf, off)[0]; off += 4
            if k == "general.alignment":
                alignment = struct.unpack_from("<I", buf, off)[0] if vt == 4 else struct.unpack_from("<Q", buf, off)[0]
                alignment = alignment or 32
            off = _skip(off, vt)

        raw_infos = []
        for _ in range(n_tensors):
            name, off = _str(off)
            n_dims = struct.unpack_from("<I", buf, off)[0]; off += 4
            dims = [struct.unpack_from("<Q", buf, off + 8 * d)[0] for d in range(n_dims)]
            off += 8 * n_dims
            ttype = struct.unpack_from("<I", buf, off)[0]; off += 4
            rel = struct.unpack_from("<Q", buf, off)[0]; off += 8
            raw_infos.append((name, dims, ttype, rel))

        data_off = ((off + alignment - 1) // alignment) * alignment
        for name, dims, ttype, rel in raw_infos:
            numel = 1
            for d in dims: numel *= d
            if ttype == 42:  # PRISM_Q2_0 (128 elements / 34 bytes)
                nbytes = (numel // 128) * 34
            elif ttype == 0:   # FP32 (4 bytes)
                nbytes = numel * 4
            elif ttype == 1:   # FP16 (2 bytes)
                nbytes = numel * 2
            else:
                nbytes = (numel // 128) * 34 if (numel % 128 == 0) else numel
            self.tensors[name] = BonsaiTensorEntry(name, data_off + rel, nbytes, dims, ttype)
            layer_idx = parse_bonsai_layer_index(name)
            self.groups.setdefault(layer_idx, []).append(name)

        for names in self.groups.values():
            names.sort()

    def _advise(self, addr_obj, advice: int) -> None:
        try:
            if hasattr(os, "posix_madvise"):
                os.posix_madvise(addr_obj, 0, len(addr_obj), advice)
        except Exception:
            pass

    def _advise_shared(self) -> None:
        will = getattr(os, "POSIX_MADV_WILLNEED", 3)
        for name in self.groups.get(-1, []):
            e = self.tensors[name]
            if e.nbytes:
                self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], will)

    @property
    def layer_indices(self) -> List[int]:
        return sorted(i for i in self.groups if i >= 0)

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    @property
    def total_model_bytes(self) -> int:
        return sum(e.nbytes for e in self.tensors.values())

    @property
    def shared_bytes(self) -> int:
        return self._shared_bytes

    @property
    def current_resident_bytes(self) -> int:
        return self._current_bytes

    @property
    def peak_resident_bytes(self) -> int:
        return self._peak_bytes

    def _layer_bytes(self, idx: int) -> int:
        return sum(self.tensors[n].nbytes for n in self.groups.get(idx, []))

    def acquire_layer(self, idx: int) -> Dict[str, memoryview]:
        if idx not in self.groups or idx < 0:
            raise KeyError(f"Unknown layer {idx}")
        dont = getattr(os, "POSIX_MADV_DONTNEED", 4)
        will = getattr(os, "POSIX_MADV_WILLNEED", 3)
        if idx in self._resident:
            self._resident.move_to_end(idx)
        else:
            while len(self._resident) >= self.resident_layer_count:
                victim, _ = self._resident.popitem(last=False)
                self._current_bytes -= self._layer_bytes(victim)
                for n in self.groups[victim]:
                    e = self.tensors[n]
                    if e.nbytes:
                        self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], dont)
            self._resident[idx] = None
            self._current_bytes += self._layer_bytes(idx)
            self._peak_bytes = max(self._peak_bytes, self._current_bytes)
            for n in self.groups[idx]:
                e = self.tensors[n]
                if e.nbytes:
                    self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], will)

        order = self.layer_indices
        pos = order.index(idx)
        if pos + 1 < len(order):
            for n in self.groups[order[pos + 1]]:
                e = self.tensors[n]
                if e.nbytes:
                    self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], will)

        mv = memoryview(self._mm)
        return {n: mv[self.tensors[n].offset:self.tensors[n].offset + self.tensors[n].nbytes]
                for n in self.groups[idx]}

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        finally:
            try:
                self._f.close()
            except Exception:
                pass
