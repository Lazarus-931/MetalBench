"""Run `./bench <kernel>` and parse its human-readable output into a structured dict.

The harness's stdout looks like:
    sqr_mm    target=speed   score=1.428
    device      : Apple M2 (m2)  8 CPU / 8 GPU / 9 GB
    occupancy   : tg_mem=16KB  max_thr/tg=896
    correctness : ✓ correct     max_err=0.000e+00
    speedup     : 1.43× vs MLX
    kernel      : 1.266 ms  (min 1.194, mean 1.289, n=300)
    mlx ref     : 1.808 ms
    compute     : 1695.8 GFLOPS
    bandwidth   :    9.9 GB/s
    arith int.  :  170.7 FLOPs/byte
    stability   : 0.98
"""
from __future__ import annotations
import fcntl
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# GPU is a single resource — concurrent ./bench runs corrupt each other's timing.
_LOCK_PATH = Path(os.environ.get(
    "AGENT_STEEL_BENCH_LOCK",
    str(Path.home() / ".agent-steel" / "bench.lock"),
))


@contextmanager
def _gpu_bench_lock():
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, f"pid={os.getpid()} t={time.time()}\n".encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@dataclass
class BenchResult:
    kernel: str
    chip: str
    gpu_cores: int | None
    correct: bool
    max_err: float | None
    speedup: float | None
    kernel_ms: float | None        # median
    kernel_ms_min: float | None
    kernel_ms_mean: float | None
    mlx_ms: float | None
    gflops: float | None
    gbps: float | None
    arith_intensity: float | None
    stability: float | None
    tg_mem_bytes: int | None
    max_threads_per_tg: int | None
    raw_stdout: str = field(repr=False)


_RX_DEVICE = re.compile(r"device\s+:\s*(.+?)\s*$", re.M)
_RX_GPU_CORES = re.compile(r"(\d+)\s*GPU\s*cores?", re.I)
_RX_OCCUPANCY = re.compile(r"occupancy\s+:\s*tg_mem=(\d+)\s*KB\s+max_thr/tg=(\d+)")
_RX_CORRECT = re.compile(r"correctness\s+:\s*(✓\s*correct|✗\s*incorrect).*?max_err=([\d.eE+-]+)")
_RX_SPEEDUP = re.compile(r"speedup\s+:\s*([\d.eE+-]+)×")
_RX_KERNEL = re.compile(
    r"kernel\s+:\s*([\d.eE+-]+)\s*ms\s+\(min\s*([\d.eE+-]+),\s*mean\s*([\d.eE+-]+),\s*n=\d+\)"
)
_RX_MLX = re.compile(r"mlx\s+ref\s+:\s*([\d.eE+-]+)\s*ms")
_RX_GFLOPS = re.compile(r"compute\s+:\s*([\d.eE+-]+)\s*GFLOPS")
_RX_GBPS = re.compile(r"bandwidth\s+:\s*([\d.eE+-]+)\s*GB/s")
_RX_AI = re.compile(r"arith\s+int\.\s*:\s*([\d.eE+-]+)")
_RX_STABILITY = re.compile(r"stability\s+:\s*([\d.eE+-]+)")


def _f(rx: re.Pattern, s: str, group: int = 1) -> float | None:
    m = rx.search(s)
    return float(m.group(group)) if m else None


def run_bench(
    kernel: str,
    *,
    iters: int = 200,
    warmup: int = 50,
    save: bool = False,
    timeout_s: int = 120,
    cwd: Path = REPO,
) -> BenchResult:
    """Run `./bench <kernel>` and return parsed metrics. Raises on subprocess failure."""
    args = ["./bench", kernel, "--", "--iters", str(iters), "--warmup", str(warmup)]
    if not save:
        args[1:1] = []
    with _gpu_bench_lock():
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout_s
        )
    out = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench {kernel} exited {proc.returncode}\nstdout:\n{out}\nstderr:\n{proc.stderr}"
        )

    correct_m = _RX_CORRECT.search(out)
    device_m = _RX_DEVICE.search(out)
    occupancy_m = _RX_OCCUPANCY.search(out)
    chip = device_m.group(1) if device_m else "unknown"
    gc_m = _RX_GPU_CORES.search(chip)
    gpu_cores = int(gc_m.group(1)) if gc_m else None

    return BenchResult(
        kernel=kernel,
        chip=chip,
        gpu_cores=gpu_cores,
        correct=("correct" in correct_m.group(1)) if correct_m else False,
        max_err=float(correct_m.group(2)) if correct_m else None,
        speedup=_f(_RX_SPEEDUP, out),
        kernel_ms=_f(_RX_KERNEL, out, 1),
        kernel_ms_min=_f(_RX_KERNEL, out, 2),
        kernel_ms_mean=_f(_RX_KERNEL, out, 3),
        mlx_ms=_f(_RX_MLX, out),
        gflops=_f(_RX_GFLOPS, out),
        gbps=_f(_RX_GBPS, out),
        arith_intensity=_f(_RX_AI, out),
        stability=_f(_RX_STABILITY, out),
        tg_mem_bytes=int(occupancy_m.group(1)) * 1024 if occupancy_m else None,
        max_threads_per_tg=int(occupancy_m.group(2)) if occupancy_m else None,
        raw_stdout=out,
    )
