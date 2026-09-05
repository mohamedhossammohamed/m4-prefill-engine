"""
Opt-in layer-by-layer GGUF weight serving (Python mirror of
core/weights/gguf_layer_streamer.h).

Keeps only `resident_layer_count` transformer layers logically resident:
acquiring a layer prefetches the next (madvise WILLNEED) and evicts the
LRU layer beyond budget (madvise DONTNEED). The whole file stays mmap'd
(zero-copy, never read into a buffer up front); acquired views alias the
mmap. Existing codec `unpack_column` paths consume the bytes unchanged.

SCOPE: serves weight BUFFERS only (Q2_0-family layouts: 128-divisible
tensors at 34 bytes/block). It does not make the uniform-GQA transformer
stack capable of Bonsai 27B's gated-delta-net hybrid topology (48/64
layers unrepresentable -- reported, not force-fit).
"""

from __future__ import annotations

import mmap
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def parse_layer_index(name: str) -> int:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return -1
    rest = name[len(prefix):]
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
class TensorEntry:
    name: str
    offset: int  # absolute file offset of tensor data
    nbytes: int  # Q2_0-family accounted bytes (0 if layout unknown)


def _read_u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _read_u64(buf: bytes, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def _read_str(buf: bytes, off: int):
    n = _read_u64(buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8"), off + n


def _skip_value(buf: bytes, off: int, vtype: int) -> int:
    # Mirrors GGUFValueType ids in core/weights/gguf_loader.h
    if vtype in (0, 1, 7):      # UINT8, INT8, BOOL
        return off + 1
    if vtype in (2, 3):         # UINT16, INT16
        return off + 2
    if vtype in (4, 5, 6):      # UINT32, INT32, FLOAT32
        return off + 4
    if vtype in (10, 11, 12):   # UINT64, INT64, FLOAT64
        return off + 8
    if vtype == 8:              # STRING
        s, off = _read_str(buf, off)
        return off
    if vtype == 9:              # ARRAY
        item_type = _read_u32(buf, off); off += 4
        count = _read_u64(buf, off); off += 8
        for _ in range(count):
            off = _skip_value(buf, off, item_type)
        return off
    raise ValueError(f"Unknown GGUF value type {vtype}")


class LayerWeightStreamer:
    """mmap-backed, bounded-residency GGUF layer server."""

    def __init__(self, path: str | Path, resident_layer_count: int = 2) -> None:
        if resident_layer_count < 1:
            raise ValueError("resident_layer_count must be >= 1")
        self.path = Path(path)
        self.resident_layer_count = int(resident_layer_count)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors: Dict[str, TensorEntry] = {}
        self.groups: Dict[int, List[str]] = {}
        self._parse()
        self._resident: OrderedDict[int, None] = OrderedDict()
        self._shared_bytes = sum(self.tensors[n].nbytes for n in self.groups.get(-1, []))
        self._current = self._shared_bytes
        self._peak = self._current
        self._advise_shared()

    # -- parsing ------------------------------------------------------
    def _parse(self) -> None:
        buf = self._mm
        if bytes(buf[0:4]) != b"GGUF":
            raise ValueError("Bad GGUF magic")
        version = _read_u32(buf, 4)
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        n_tensors = _read_u64(buf, 8)
        n_kv = _read_u64(buf, 16)
        alignment = 32
        off = 24
        for _ in range(n_kv):
            key, off = _read_str(buf, off)
            vtype = _read_u32(buf, off); off += 4
            if key == "general.alignment":
                if vtype in (4, 10):
                    alignment = _read_u32(buf, off) if vtype == 4 else _read_u64(buf, off)
                    alignment = alignment or 32
            off = _skip_value(buf, off, vtype)
        raw_infos = []
        for _ in range(n_tensors):
            name, off = _read_str(buf, off)
            n_dims = _read_u32(buf, off); off += 4
            dims = [_read_u64(buf, off + 8 * d) for d in range(n_dims)]
            off += 8 * n_dims
            gtype = _read_u32(buf, off); off += 4
            rel = _read_u64(buf, off); off += 8
            raw_infos.append((name, dims, rel))
        data_off = ((off + alignment - 1) // alignment) * alignment
        for name, dims, rel in raw_infos:
            numel = 1
            for d in dims:
                numel *= d
            nbytes = (numel // 128) * 34 if dims and numel % 128 == 0 else 0
            self.tensors[name] = TensorEntry(name, data_off + rel, nbytes)
            self.groups.setdefault(parse_layer_index(name), []).append(name)
        for names in self.groups.values():
            names.sort()

    # -- madvise helpers (best-effort) --------------------------------
    @staticmethod
    def _advise(addr_obj, advice: int) -> None:
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

    # -- serving ------------------------------------------------------
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
        return self._current

    @property
    def peak_resident_bytes(self) -> int:
        return self._peak

    def _layer_bytes(self, idx: int) -> int:
        return sum(self.tensors[n].nbytes for n in self.groups.get(idx, []))

    def acquire_layer(self, idx: int) -> Dict[str, memoryview]:
        """Makes layer `idx` resident; returns {tensor_name: bytes view}."""
        if idx not in self.groups or idx < 0:
            raise KeyError(f"Unknown layer {idx}")
        dont = getattr(os, "POSIX_MADV_DONTNEED", 4)
        will = getattr(os, "POSIX_MADV_WILLNEED", 3)
        if idx in self._resident:
            self._resident.move_to_end(idx)
        else:
            while len(self._resident) >= self.resident_layer_count:
                victim, _ = self._resident.popitem(last=False)
                self._current -= self._layer_bytes(victim)
                for n in self.groups[victim]:
                    e = self.tensors[n]
                    if e.nbytes:
                        self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], dont)
            self._resident[idx] = None
            self._current += self._layer_bytes(idx)
            self._peak = max(self._peak, self._current)
            for n in self.groups[idx]:
                e = self.tensors[n]
                if e.nbytes:
                    self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], will)
        # Prefetch next layer (advisory, uncounted).
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

    def full_resident_bytes(self, name: str) -> bytes:
        """Reference path: raw bytes straight from mmap (for parity checks)."""
        e = self.tensors[name]
        return bytes(self._mm[e.offset:e.offset + e.nbytes])

    def close(self) -> None:
        try:
            self._mm.close()
        except BufferError:
            # Caller still holds exported views; release them then retry.
            import gc
            gc.collect()
            try:
                self._mm.close()
            except BufferError:
                pass
        finally:
            try:
                self._f.close()
            except Exception:
                pass

    def __enter__(self) -> "LayerWeightStreamer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_synthetic_gguf(path: str | Path, num_layers: int = 8,
                          K: int = 256, N: int = 128, alignment: int = 32) -> List[str]:
    """Writes a synthetic multi-layer Q2_0 GGUF fixture (test/demo use)."""
    names: List[str] = []
    for lyr in range(num_layers):
        names.append(f"model.layers.{lyr}.attn.q.weight")
        names.append(f"model.layers.{lyr}.mlp.gate.weight")
    names.append("model.embed.weight")

    def _u32(v: int) -> bytes:
        return struct.pack("<I", v)

    def _u64(v: int) -> bytes:
        return struct.pack("<Q", v)

    def _s(s: str) -> bytes:
        b = s.encode()
        return _u64(len(b)) + b

    buf = bytearray(b"GGUF" + _u32(3) + _u64(len(names)) + _u64(2))
    buf += _s("general.architecture") + _u32(8) + _s("prism")
    buf += _s("general.alignment") + _u32(4) + _u32(alignment)

    per_block, dims_of = 34, {}
    for n in names:
        nn = (N // 2) if n == "model.embed.weight" else N
        dims_of[n] = (K, nn)
    cursor = 0
    offs = {}
    for n in names:
        Kk, Nn = dims_of[n]
        buf += _s(n) + _u32(2) + _u64(Kk) + _u64(Nn) + _u32(100)
        aligned = ((cursor + alignment - 1) // alignment) * alignment
        buf += _u64(aligned)
        size = (Kk * Nn // 128) * per_block
        offs[n] = (aligned, size)
        cursor = aligned + size
    data_off = ((len(buf) + alignment - 1) // alignment) * alignment
    buf += bytes(data_off - len(buf))
    data_start = len(buf)
    region = ((cursor + alignment - 1) // alignment) * alignment
    buf += bytes(region)
    for i, n in enumerate(names):
        Kk, Nn = dims_of[n]
        size = (Kk * Nn // 128) * per_block
        seed = (0xA0 if n == "model.embed.weight" else (0x10 + i * 0x11)) & 0xFF
        blk = bytearray()
        nblocks = size // per_block
        for b in range(nblocks):
            blk += struct.pack("<H", 0x2C00)  # FP16 0.0625 LE
            blk += bytes((seed ^ ((j * 7 + b) & 0xFF)) for j in range(32))
        start = data_start + offs[n][0]
        buf[start:start + size] = blk
    Path(path).write_bytes(bytes(buf))
    return names
