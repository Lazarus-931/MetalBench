"""Smoke asserts for `./bench`. No GPU required."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_bench_script_is_executable():
    p = REPO / "bench"
    assert p.is_file() and (p.stat().st_mode & 0o111)


def test_bench_help_runs():
    r = subprocess.run(["./bench", "--help"], cwd=REPO,
                       capture_output=True, text=True, timeout=10)
    assert "usage" in (r.stdout + r.stderr).lower()


def test_session_json_parses():
    assert isinstance(json.loads((REPO / "session.json").read_text()), dict)


@pytest.mark.parametrize("set_name", ["common", "standard", "full"])
def test_registry_nonempty(set_name):
    reg = REPO / "mlx" / "kernels" / set_name / "registry.py"
    if not reg.is_file():
        pytest.skip(f"no {set_name} registry")
    ns: dict = {}
    exec(reg.read_text(), ns)
    assert ns.get("REGISTRY"), f"{set_name}/REGISTRY empty"
