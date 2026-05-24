"""Smoke tests for the chips.json single source of truth.

These tests guard the chip-extensibility refactor: every legacy hardcoded
chip list elsewhere in the codebase must be reproducible from the registry,
so adding a new generation (M6, M7, ...) requires only a chips.json edit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_steel import chips  # noqa: E402


def test_detect_generation_basic():
    assert chips.detect_generation("Apple M5 Max") == "m5"
    assert chips.detect_generation("Apple M2 Ultra") == "m2"
    assert chips.detect_generation("Apple M1") == "m1"
    assert chips.detect_generation("Apple M4 Pro") == "m4"


def test_detect_generation_fallback():
    assert chips.detect_generation(None) == chips.DEFAULT_FALLBACK_GEN
    assert chips.detect_generation("") == chips.DEFAULT_FALLBACK_GEN
    assert chips.detect_generation("intel xeon", fallback="m1") == "m1"


def test_detect_variant():
    assert chips.detect_variant("Apple M2 Max") == "m2_max"
    assert chips.detect_variant("Apple M5 Ultra") == "m5_ultra"
    assert chips.detect_variant("Apple M3") == "m3"
    assert chips.detect_variant("Apple M4 Pro") == "m4_pro"
    assert chips.detect_variant(None) == "unknown"


def test_list_generations_newest_first():
    gens = chips.list_generations()
    assert gens[0] == "m5"
    assert gens[-1] == "m1"
    assert set(gens) >= {"m1", "m2", "m3", "m4", "m5"}


def test_list_chip_types_matches_legacy_layout():
    """Reproduces the old mlx_helpers._CHIP_TYPES tuple shape exactly:
    20 entries, ordered most-specific-first within each generation, newest gen first."""
    pairs = chips.list_chip_types()
    assert len(pairs) == 5 * 4  # 5 generations x 4 variants
    # First quadruplet must be the newest generation (M5), Ultra first.
    assert pairs[0] == ("M5 Ultra", "m5_ultra")
    assert pairs[1] == ("M5 Max", "m5_max")
    assert pairs[2] == ("M5 Pro", "m5_pro")
    assert pairs[3] == ("M5", "m5")
    # Last quadruplet must be M1, bare last.
    assert pairs[-4] == ("M1 Ultra", "m1_ultra")
    assert pairs[-1] == ("M1", "m1")


def test_peak_table_matches_legacy_values():
    """Old roofline.CHIP_PEAKS values must be reproducible from registry."""
    legacy = {
        "m1": dict(bw_GBps=68.25, compute_TFLOPS=2.6),
        "m2": dict(bw_GBps=100.0, compute_TFLOPS=3.6),
        "m3": dict(bw_GBps=102.4, compute_TFLOPS=4.1),
        "m4": dict(bw_GBps=120.0, compute_TFLOPS=4.5),
        "m5": dict(bw_GBps=150.0, compute_TFLOPS=5.5),
    }
    for gen, exp in legacy.items():
        assert chips.peak(gen) == exp


def test_ceiling_table_matches_legacy_values():
    """Old gputrace_check._CHIP_CEILINGS values must be reproducible."""
    for gen in ("m1", "m2", "m3", "m4", "m5"):
        c = chips.ceiling(gen)
        assert c["tg_memory_max_bytes"] == 32768
        assert c["peak_compute_TFLOPS"] > 0
        assert c["peak_bandwidth_GBps"] > 0


def test_chip_table_h_is_in_sync():
    """The generated C++ header must contain every variant in chips.json."""
    header = (REPO / "metal" / "scripts" / "chip_table.h").read_text(encoding="utf-8")
    for needle, tag in chips.list_chip_types():
        assert f'"{tag}"' in header, f"missing tag {tag} in chip_table.h"
        assert f'"{needle}"' in header, f"missing needle {needle!r} in chip_table.h"


if __name__ == "__main__":
    # Plain-python entry point so this can run without pytest.
    import inspect
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and inspect.getmodule(fn).__name__ == "__main__":
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failed else 0)
