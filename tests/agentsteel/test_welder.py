"""Welder unit tests — pure-Python, no LLM calls."""
from __future__ import annotations
from pathlib import Path

from agent_steel.welder.agent import (
    _parse_json,
    _read_sample_registry,
    _resolve_paths,
    _append_registry_entry,
)

REPO = Path(__file__).resolve().parents[2]


def test_parse_json_strips_fences():
    text = '```json\n{"new_metal_source": "x", "change_summary": "y"}\n```'
    out = _parse_json(text)
    assert out["new_metal_source"] == "x"


def test_parse_json_bare():
    out = _parse_json('{"a": 1}')
    assert out == {"a": 1}


def test_read_sample_registry_returns_content():
    s = _read_sample_registry("common")
    assert "REGISTRY" in s


def test_resolve_paths():
    mlx, reg, metal, km = _resolve_paths("relu", "common")
    assert mlx.name == "relu.py" and "common" in str(mlx)
    assert reg.name == "registry.py" and "common" in str(reg)
    assert metal.name == "relu.metal" and "common" in str(metal)
    assert km.name == "KERNELS.md"


def test_append_registry_idempotent(tmp_path):
    reg = tmp_path / "registry.py"
    reg.write_text("REGISTRY = {}\n")
    _append_registry_entry(reg, "fakekernel", 'REGISTRY["fakekernel"] = dict(x=1)')
    after1 = reg.read_text()
    assert "fakekernel" in after1
    _append_registry_entry(reg, "fakekernel", 'REGISTRY["fakekernel"] = dict(x=2)')
    after2 = reg.read_text()
    assert after1 == after2  # no-op on second call
