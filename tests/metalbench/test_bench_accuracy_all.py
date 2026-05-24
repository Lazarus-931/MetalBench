"""Run every registered MetalBench kernel and verify Metal-vs-MLX accuracy ≤ 1%.

Heavy. Skipped unless METALBENCH_FULL=1 or running on macOS with Metal.
Requires a successful `make all` build.
"""
from __future__ import annotations
import os
import platform
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RTOL = 0.05  # 5% accuracy bar


def _discover_kernels() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in ("common", "standard", "full"):
        reg = REPO / "mlx" / "kernels" / s / "registry.py"
        if not reg.is_file():
            continue
        ns: dict = {}
        exec(reg.read_text(), ns)
        for k in sorted(ns.get("REGISTRY", {})):
            out.append((s, k))
    return out


_RX_CORRECT = re.compile(r"correctness\s+:\s*(✓|✗).*?max_err=([\d.eE+-]+)")


def _bench_correctness(kernel: str) -> tuple[bool, float]:
    r = subprocess.run(
        ["./bench", kernel, "--no-save", "--", "--iters", "5", "--warmup", "2"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    m = _RX_CORRECT.search(r.stdout + r.stderr)
    if not m:
        return False, float("inf")
    return m.group(1) == "✓", float(m.group(2))


_RUN_FULL = os.environ.get("METALBENCH_FULL") == "1" or platform.system() == "Darwin"


@pytest.mark.skipif(not _RUN_FULL, reason="set METALBENCH_FULL=1 or run on macOS")
@pytest.mark.parametrize("set_name,kernel", _discover_kernels())
def test_kernel_accuracy_within_5pct(set_name, kernel):
    ok, err = _bench_correctness(kernel)
    assert ok, f"{set_name}/{kernel}: correctness failed (max_err={err})"
    assert err <= RTOL, f"{set_name}/{kernel}: max_err={err:.4e} > {RTOL}"
