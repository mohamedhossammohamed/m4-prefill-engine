"""
Weight-streamer verification: bounded residency + bit-identical bytes.
Run: .venv/bin/python tests/test_weight_streamer.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.weight_streamer import LayerWeightStreamer, build_synthetic_gguf, parse_layer_index


def test_parse():
    assert parse_layer_index("model.layers.0.attn.q.weight") == 0
    assert parse_layer_index("model.layers.12.mlp.gate.weight") == 12
    assert parse_layer_index("model.embed.weight") == -1
    print("[PASS] layer index parsing")


def test_bounded_and_identical():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "multi.gguf")
        names = build_synthetic_gguf(path, num_layers=4, K=256, N=128)
        with LayerWeightStreamer(path, resident_layer_count=1) as st:
            assert st.num_layers == 4, st.num_layers
            per_layer = (256 * 128 // 128) * 34 * 2
            assert st.total_model_bytes == per_layer * 4 + st.shared_bytes
            checked = 0
            for lyr in st.layer_indices:
                views = st.acquire_layer(lyr)
                for n, v in views.items():
                    assert bytes(v) == st.full_resident_bytes(n), f"mismatch {n}"
                    checked += 1
                del views
            assert checked == 8, checked
            assert st.total_model_bytes > st.peak_resident_bytes * 2, st.peak_resident_bytes
            assert st.peak_resident_bytes <= per_layer + st.shared_bytes, (
                st.peak_resident_bytes, per_layer + st.shared_bytes)
            print(f"[PASS] bounded peak={st.peak_resident_bytes}B total={st.total_model_bytes}B "
                  f"({checked} tensors identical)")
            try:
                st.acquire_layer(999)
                raise SystemExit("expected KeyError for unknown layer")
            except KeyError:
                print("[PASS] unknown-layer guard")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TEST SUITE: WEIGHT STREAMER (Task 1, Python mirror)")
    print("=" * 80)
    test_parse()
    test_bounded_and_identical()
    print("=" * 80)
    print("ALL WEIGHT STREAMER TESTS PASSED (EXIT CODE 0)")
    print("=" * 80)
