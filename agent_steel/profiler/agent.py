"""Profiler agent — gputrace + bench → 2-3 paragraph narrative.

One LLM call. No suggested-edits, no ranked candidates, no big packet.
Downstream Optimizer reads the narrative + AttemptDB log to decide what to try.

    from agent_steel.profiler import profile
    result = profile("relu", provider=p, gputrace_path="results/m2/relu.gputrace")
    print(result.narrative)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..providers import Message, Provider
from .bench_runner import BenchResult, run_bench
from .chip_metrics import derive_metrics
from .gputrace import parse as parse_gputrace

REPO = Path(__file__).resolve().parents[2]
_PROMPT_PATH = REPO / "agent_steel" / "prompts" / "profiler.md"


@dataclass
class ProfileResult:
    """Slim Profiler output. The narrative is the primary product."""
    kernel: str
    chip: str                                       # raw brand string
    chip_generation: str                            # "m2" / "m4" / ...
    chip_variant: str                               # "base" / "pro" / "max" / "ultra"
    narrative: str                                  # 2-3 paragraphs for the Optimizer
    bench: BenchResult = field(repr=False)
    chip_aware_metrics: dict[str, Any] | None = field(default=None, repr=False)


def _identify_chip(chip_str: str) -> tuple[str, str]:
    s = (chip_str or "").lower()
    gen = ("m5" if "m5" in s
           else "m4" if "m4" in s
           else "m3" if "m3" in s
           else "m1" if "m1" in s
           else "m2")
    variant = ("ultra" if "ultra" in s
               else "max" if "max" in s
               else "pro" if "pro" in s
               else "base")
    return gen, variant


def _load_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text()
    return (
        "You are a GPU profiler. Read the measurements below and write a 2-3 "
        "paragraph summary of what the GPU did and what the bottleneck is."
    )


def _build_user_message(
    kernel: str,
    bench: BenchResult,
    chip_aware: dict[str, Any] | None,
) -> str:
    payload = {
        "kernel": kernel,
        "chip": bench.chip,
        "bench": {
            "kernel_ms_median": bench.kernel_ms,
            "kernel_ms_min": bench.kernel_ms_min,
            "kernel_ms_mean": bench.kernel_ms_mean,
            "stability": bench.stability,
            "GFLOPS": bench.gflops,
            "BW_GBps": bench.gbps,
            "arith_intensity": bench.arith_intensity,
            "speedup_vs_mlx": bench.speedup,
            "tg_mem_bytes": bench.tg_mem_bytes,
            "max_threads_per_tg": bench.max_threads_per_tg,
            "correct": bench.correct,
            "max_err": bench.max_err,
        },
        "chip_aware_metrics": chip_aware,
    }
    return (
        "Write a 2-3 paragraph profile summary for this kernel based on the "
        "measurements below. Output prose only — no headers, no bullet lists, "
        "no JSON.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def profile(
    kernel: str,
    *,
    provider: Provider,
    gputrace_path: str | None = None,
) -> ProfileResult:
    """Bench the kernel, parse its .gputrace if provided, ask the LLM for a
    2-3 paragraph summary. Returns ProfileResult."""
    bench = run_bench(kernel)
    chip_gen, chip_variant = _identify_chip(bench.chip)

    chip_aware: dict[str, Any] | None = None
    if gputrace_path:
        try:
            parsed = parse_gputrace(gputrace_path)
            chip_aware = derive_metrics(
                bench={
                    "kernel_ms": bench.kernel_ms,
                    "flops": (bench.kernel_ms or 0) * (bench.gflops or 0) * 1e6,
                    "bytes": (bench.kernel_ms or 0) * (bench.gbps or 0) * 1e6,
                    "detected_gpu_cores": bench.gpu_cores,
                },
                parsed_trace=parsed,
                generation=chip_gen,
                variant=chip_variant,
            )
        except Exception as e:
            chip_aware = {"_derive_error": str(e)}

    resp = provider.generate(
        [
            Message("system", _load_system_prompt()),
            Message("user", _build_user_message(kernel, bench, chip_aware)),
        ],
        max_tokens=600,
        temperature=0.2,
    )

    return ProfileResult(
        kernel=kernel,
        chip=bench.chip,
        chip_generation=chip_gen,
        chip_variant=chip_variant,
        narrative=resp.text.strip(),
        bench=bench,
        chip_aware_metrics=chip_aware,
    )


class ProfilerAgent:
    """OO wrapper for symmetry with Optimizer/Verifier."""

    def __init__(self, provider: Provider):
        self.provider = provider

    def run(self, kernel: str, *, gputrace_path: str | None = None) -> ProfileResult:
        return profile(kernel, provider=self.provider, gputrace_path=gputrace_path)
