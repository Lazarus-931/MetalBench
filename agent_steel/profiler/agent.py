"""Profiler Agent — diagnostic stage of the Agent Steel pipeline.

Job: take a kernel name, produce a structured `ProfilerReport` that the
downstream Optimizer agent uses to pick its next attempt.

Design (see design.md / chat thread):
- Deterministic pass — uses `mlx/scripts/roofline.py` classification we
  already produce. No LLM call needed for the classification itself.
- LLM pass — bridges the numbers to the .metal source. Reads the kernel
  source, the MLX reference, the registry entry, and explains *why* the
  kernel is at that SOL with line-level pointers to improve.

The LLM never overrides the roofline classification — numbers do that.
"""
from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .bench_runner import BenchResult, run_bench
from ..providers import Message, Provider, get_provider

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Locate source files for a kernel.
# ---------------------------------------------------------------------------

def _find_metal_source(kernel: str) -> tuple[str, str]:
    """Return (set_name, source_text). Picks dir/default.metal if dir, else flat."""
    for s in ("common", "standard", "full"):
        flat = REPO / "metal" / "kernels" / s / f"{kernel}.metal"
        dir_default = REPO / "metal" / "kernels" / s / kernel / "default.metal"
        # Prefer the chip-agnostic 'default.metal' if it's split; if not split,
        # the flat file is the source of truth. Chip variants (m4.metal etc.)
        # are derivatives — the optimizer can read them separately if needed.
        if dir_default.is_file():
            return s, dir_default.read_text()
        if flat.is_file():
            return s, flat.read_text()
        # No default but a chip variant exists — e.g. huber_loss/m2.metal alone.
        d = REPO / "metal" / "kernels" / s / kernel
        if d.is_dir():
            for v in sorted(d.iterdir()):
                if v.suffix == ".metal":
                    return s, v.read_text()
    raise FileNotFoundError(f"no .metal source found for kernel {kernel!r}")


def _find_mlx_reference(kernel: str, set_name: str) -> str:
    p = REPO / "mlx" / "kernels" / set_name / f"{kernel}.py"
    return p.read_text() if p.is_file() else ""


def _find_registry_entry(kernel: str, set_name: str) -> str:
    """Return the raw text of `REGISTRY["<kernel>"] = dict(...)` block.

    Cheap regex-based extraction. The LLM only needs to see the dict body,
    not the helper functions surrounding it.
    """
    p = REPO / "mlx" / "kernels" / set_name / "registry.py"
    if not p.is_file():
        return ""
    text = p.read_text()
    m = re.search(
        rf'(?:^\s*REGISTRY\[["\']({re.escape(kernel)})["\']\]\s*=\s*dict\(.*?\n\)|'
        rf"^ew\(.*?{re.escape(kernel)}.*?\)\s*$)",
        text,
        re.M | re.S,
    )
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Read the session.json record for this kernel (gives us the saved-best metrics
# including the roofline classification when the harness emits one).
# ---------------------------------------------------------------------------

def _session_entry(kernel: str, chip_id: str) -> dict[str, Any] | None:
    p = REPO / "session.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text())
    return s.get(chip_id, {}).get(kernel)


# ---------------------------------------------------------------------------
# Roofline classification — call the existing module so we never re-implement.
# ---------------------------------------------------------------------------

def _classify_roofline(bench: BenchResult, registry_entry: str) -> dict[str, Any]:
    """Use `mlx/scripts/roofline.py` to classify. Returns the dict it produces."""
    # roofline.classify needs (chip_type, flops, bytes_, median_ms). We get
    # flops/bytes from the registry entry (re.search the literal int).
    flops_m = re.search(r"flops\s*=\s*([0-9_*+\- ]+)", registry_entry)
    bytes_m = re.search(r"bytes\s*=\s*([0-9_*+\- 4]+)", registry_entry)
    flops = eval(flops_m.group(1)) if flops_m else 0.0
    bytes_ = eval(bytes_m.group(1)) if bytes_m else 0.0

    from agent_steel.chips import detect_generation
    chip_type = detect_generation(bench.chip, fallback="m2")

    sys.path.insert(0, str(REPO / "mlx" / "scripts"))
    try:
        import roofline  # type: ignore
        return roofline.classify(chip_type, flops, bytes_, bench.kernel_ms or 0.001)
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# The packet that gets shipped to the LLM.
# ---------------------------------------------------------------------------

def _registry_text_to_dict(text: str) -> dict[str, Any]:
    """Parse the relevant fields out of a `REGISTRY["<name>"] = dict(...)` block
    into a Python dict for `gputrace_check`. Best-effort — only the fields we
    care about are extracted, anything missing returns None.
    """
    out: dict[str, Any] = {}
    for key in ("metal_function", "grid", "threadgroup", "input_shapes",
                "flops", "bytes"):
        m = re.search(rf"{key}\s*=\s*(.*?)(?=,\s*\n|\s*\)\s*$)", text, re.S)
        if not m:
            continue
        raw = m.group(1).strip().rstrip(",")
        # `eval` here is acceptable — we control the registry text and only
        # accept arithmetic / list / tuple / string literals.
        try:
            out[key] = eval(raw)
        except Exception:
            out[key] = raw
    return out


def _build_packet(
    kernel: str,
    set_name: str,
    bench: BenchResult,
    roofline: dict[str, Any],
    metal_source: str,
    mlx_reference: str,
    registry_entry: str,
    session_record: dict[str, Any] | None,
    prior_attempts: list[str],
    gputrace_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kernel": kernel,
        "set": set_name,
        "chip": bench.chip,
        "correct": bench.correct,
        "max_err": bench.max_err,
        "roofline": {
            "classification": roofline.get("classification"),
            "sol": roofline.get("sol"),
            "sol_compute": roofline.get("sol_compute"),
            "sol_memory": roofline.get("sol_memory"),
            "arith_intensity": roofline.get("intensity"),
            "ridge_intensity": roofline.get("ridge"),
            "gflops": roofline.get("gflops"),
            "gbps": roofline.get("gbps"),
            "peak_compute_TFLOPS": roofline.get("peak", {}).get("compute_TFLOPS"),
            "peak_bandwidth_GBps": roofline.get("peak", {}).get("bw_GBps"),
            "canned_suggestion": roofline.get("suggest"),
        },
        "timing": {
            "median_ms": bench.kernel_ms,
            "min_ms": bench.kernel_ms_min,
            "mean_ms": bench.kernel_ms_mean,
            "stability": bench.stability,
            "speedup_vs_mlx": bench.speedup,
            "mlx_median_ms": bench.mlx_ms,
        },
        "occupancy": {
            "tg_mem_bytes": bench.tg_mem_bytes,
            "max_threads_per_tg": bench.max_threads_per_tg,
        },
        "metal_source": metal_source,
        "mlx_reference": mlx_reference,
        "registry_entry": registry_entry,
        "session_record": session_record,
        "prior_attempts": prior_attempts,
        "gputrace": gputrace_summary,
    }


# ---------------------------------------------------------------------------
# Output type.
# ---------------------------------------------------------------------------

@dataclass
class SuggestedEdit:
    technique: str          # e.g. "simdgroup_matrix MMA on QK^T"
    rationale: str          # why this should help, citing the roofline + source
    target_lines: str       # e.g. "metal/.../softmax_attention.metal lines 80–120"
    expected_impact: str    # e.g. "could lift SOL_compute from 13% to ~40%"


@dataclass
class ProfilerReport:
    kernel: str
    chip: str
    bottleneck_class: str   # passthrough from roofline.classification
    sol: float              # passthrough from roofline.sol
    confidence: float       # LLM-reported, 0–1
    code_analysis: str      # 2–4 sentences explaining the SOL given the source
    suggested_edits: list[SuggestedEdit]
    packet: dict[str, Any] = field(repr=False)      # full diagnostic packet
    raw_llm_response: str = field(repr=False, default="")


# ---------------------------------------------------------------------------
# The prompt sent to the LLM.
# ---------------------------------------------------------------------------

_PROMPT_PATH = REPO / "agent_steel" / "prompts" / "profiler.md"


def _load_system_prompt() -> str:
    """Read the profiler system prompt from `prompts/profiler.md`.

    Falls back to a minimal inline prompt if the file is missing — that
    prevents agent_steel from breaking if the prompts dir gets relocated.
    """
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text()
    return (
        "You are a kernel-performance analyst. Return JSON with "
        "code_analysis (string), confidence (0-1), and suggested_edits (array)."
    )


def _make_user_message(packet: dict[str, Any]) -> str:
    # The packet is dumped as JSON. metal_source/mlx_reference are big strings,
    # but smaller than the model's context for any single kernel.
    return (
        "Diagnose this kernel and return JSON per the system instructions.\n\n"
        + json.dumps(packet, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Main entry.
# ---------------------------------------------------------------------------

def profile(
    kernel: str,
    *,
    provider: Provider | None = None,
    prior_attempts: list[str] | None = None,
    bench: BenchResult | None = None,
    skip_llm: bool = False,
    gputrace_path: str | None = None,
) -> ProfilerReport:
    """Profile one kernel. Optionally pre-supply a BenchResult to skip the rerun,
    or a `.gputrace` path to enrich the diagnostic with parsed dispatch info."""
    if bench is None:
        bench = run_bench(kernel)

    set_name, metal_source = _find_metal_source(kernel)
    mlx_reference = _find_mlx_reference(kernel, set_name)
    registry_entry = _find_registry_entry(kernel, set_name)
    roofline = _classify_roofline(bench, registry_entry)

    from agent_steel.chips import detect_generation
    chip_id = "apple-" + detect_generation(bench.chip, fallback="m1")
    session_record = _session_entry(kernel, chip_id)

    # gputrace cross-check — uses gputrace_check.py to produce a
    # pre-computed dispatch/buffer correctness diagnostic. Always called when
    # we have a bench result; the parsed-trace cross-check is added only when
    # a trace path is supplied.
    gputrace_summary = None
    try:
        from .gputrace_check import gputrace_check
        registry_dict = _registry_text_to_dict(registry_entry)
        gputrace_summary = gputrace_check(
            trace_path=gputrace_path,
            registry_entry=registry_dict,
            bench=bench,
            metal_source=metal_source,
            session_record=session_record,
        )
    except Exception as e:
        gputrace_summary = {"_check_error": str(e)}

    packet = _build_packet(
        kernel, set_name, bench, roofline,
        metal_source, mlx_reference, registry_entry,
        session_record, prior_attempts or [],
        gputrace_summary=gputrace_summary,
    )

    # Deterministic fast-exit cases: don't burn LLM tokens.
    if not bench.correct:
        return ProfilerReport(
            kernel=kernel, chip=chip_id,
            bottleneck_class="correctness_failure",
            sol=0.0, confidence=1.0,
            code_analysis=f"Kernel fails correctness (max_err={bench.max_err}). Fix correctness before any optimization work.",
            suggested_edits=[SuggestedEdit(
                technique="fix correctness",
                rationale="Optimization is meaningless on incorrect output.",
                target_lines="entire .metal file",
                expected_impact="restores correctness; speedup gating becomes valid",
            )],
            packet=packet,
        )

    sol = float(roofline.get("sol") or 0.0)
    if sol >= 0.85:
        return ProfilerReport(
            kernel=kernel, chip=chip_id,
            bottleneck_class=roofline.get("classification") or "near_optimal",
            sol=sol, confidence=0.95,
            code_analysis=f"Kernel is at {sol*100:.0f}% of speed-of-light — near optimal for this chip on this workload. Further changes likely regress; don't optimize.",
            suggested_edits=[],
            packet=packet,
        )

    if skip_llm or provider is None:
        # Deterministic-only mode: return the canned roofline suggestion.
        return ProfilerReport(
            kernel=kernel, chip=chip_id,
            bottleneck_class=roofline.get("classification") or "unknown",
            sol=sol, confidence=0.6,
            code_analysis=f"Roofline says {roofline.get('classification')} at sol={sol*100:.0f}%. (LLM analysis skipped.)",
            suggested_edits=[SuggestedEdit(
                technique=roofline.get("suggest", ""),
                rationale="Canned suggestion from roofline.py for this bottleneck class.",
                target_lines="(unspecified — enable LLM for line-level pointers)",
                expected_impact="unknown without code analysis",
            )],
            packet=packet,
        )

    # LLM pass.
    resp = provider.generate(
        [
            Message("system", _load_system_prompt()),
            Message("user", _make_user_message(packet)),
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    parsed = _parse_llm_json(resp.text)
    return ProfilerReport(
        kernel=kernel,
        chip=chip_id,
        bottleneck_class=roofline.get("classification") or "unknown",
        sol=sol,
        confidence=float(parsed.get("confidence", 0.5)),
        code_analysis=parsed.get("code_analysis", ""),
        suggested_edits=[SuggestedEdit(**e) for e in parsed.get("suggested_edits", [])],
        packet=packet,
        raw_llm_response=resp.text,
    )


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Strip code fences if present, parse JSON. Returns empty dict on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Best-effort: extract the largest {...} block.
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# CLI: python -m agent_steel.profiler_agent <kernel> [--no-llm]
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="profiler_agent")
    ap.add_argument("kernel")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM call, use canned roofline suggestion only")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None,
                    help="Override provider's default model")
    ap.add_argument("--output", choices=["json", "text"], default="text")
    args = ap.parse_args(argv)

    p = None if args.no_llm else get_provider(args.provider, default_model=args.model)
    report = profile(args.kernel, provider=p, skip_llm=args.no_llm)

    if args.output == "json":
        print(json.dumps({
            "kernel": report.kernel,
            "chip": report.chip,
            "bottleneck_class": report.bottleneck_class,
            "sol": report.sol,
            "confidence": report.confidence,
            "code_analysis": report.code_analysis,
            "suggested_edits": [asdict(e) for e in report.suggested_edits],
        }, indent=2))
    else:
        print(f"\n  kernel       : {report.kernel}  ({report.chip})")
        print(f"  bottleneck   : {report.bottleneck_class}")
        print(f"  sol          : {report.sol*100:.0f}%")
        print(f"  confidence   : {report.confidence:.2f}")
        print(f"\n  analysis     : {report.code_analysis}\n")
        if report.suggested_edits:
            print("  suggested edits (ranked):")
            for i, e in enumerate(report.suggested_edits, 1):
                print(f"    {i}. {e.technique}")
                print(f"       why: {e.rationale}")
                print(f"       where: {e.target_lines}")
                print(f"       impact: {e.expected_impact}\n")
        else:
            print("  no edits suggested\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
