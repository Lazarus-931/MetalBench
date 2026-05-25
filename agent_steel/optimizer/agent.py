"""Optimizer agent — LLM that writes the next kernel from a Profiler narrative.

Inputs:
- ProfileResult.narrative (2-3 paragraph diagnosis from the Profiler)
- AttemptDB log for (kernel, chip)
- Current .metal source (the on-disk best)
- MLX reference (the ground-truth spec)

Outputs (OptimizerResult):
- new .metal text written to optimizer/staging/<kernel>.metal
- 2-3 sentence change_summary the LLM wrote
- accuracy_passed: True iff bench(staging) returns correct=True
- BenchResult of the staging run (perf comes from Verifier; we only gate on correctness here)

Promotion (staging → metal/kernels/<set>/<kernel>.metal) is the Loop's
responsibility, not the Optimizer's. The Optimizer never mutates committed
kernel sources; it only stages.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..history import AttemptDB, AttemptEntry
from ..profiler import ProfileResult, run_bench
from ..providers import Message, Provider

REPO = Path(__file__).resolve().parents[2]
STAGING_DIR = REPO / "agent_steel" / "optimizer" / "staging"
_PROMPT_PATH = REPO / "agent_steel" / "prompts" / "optimizer.md"


@dataclass
class OptimizerResult:
    kernel: str
    chip: str                                       # apple-m2 / apple-m4
    new_metal_source: str = field(repr=False)       # the candidate kernel
    change_summary: str = ""                        # 2-3 sentence prose
    staging_path: Path | None = None                # where it was staged
    accuracy_passed: bool = False
    accuracy_max_err: float | None = None
    accuracy_kernel_ms: float | None = None         # perf from accuracy run (informational only)
    notes: str = ""


def _resolve_metal_path(kernel: str, chip_generation: str) -> tuple[Path, str]:
    """Return (active_metal_path, set_name). Handles flat .metal and per-chip dirs."""
    for set_name in ("common", "standard", "full"):
        flat = REPO / "metal" / "kernels" / set_name / f"{kernel}.metal"
        if flat.is_file():
            return flat, set_name
        d = REPO / "metal" / "kernels" / set_name / kernel
        if d.is_dir():
            cand = d / f"{chip_generation}.metal"
            if cand.is_file():
                return cand, set_name
            default = d / "default.metal"
            if default.is_file():
                return default, set_name
            for f in sorted(d.iterdir()):
                if f.suffix == ".metal":
                    return f, set_name
    raise FileNotFoundError(f"no .metal file for kernel {kernel!r}")


def _resolve_mlx_path(kernel: str, set_name: str) -> Path:
    return REPO / "mlx" / "kernels" / set_name / f"{kernel}.py"


def _render_attempt_log(db: AttemptDB, kernel: str, chip: str, limit: int = 10) -> str:
    """Markdown table of the most-recent attempts on this kernel × chip."""
    entries = list(db.read(kernel, chip))
    if not entries:
        return "_(no prior attempts on this kernel × chip)_"
    recent = entries[-limit:]
    lines = [
        "| # | technique | before→after ms | Δ% | kept | reason |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(recent, start=max(1, len(entries) - limit + 1)):
        delta = f"{e.improvement_pct:+.1f}%" if e.improvement_pct is not None else "—"
        before = f"{e.before_ms:.4f}" if e.before_ms is not None else "—"
        after = f"{e.after_ms:.4f}" if e.after_ms is not None else "—"
        kept = "✓" if e.kept else "✗"
        reason = e.rollback_reason or ("kept" if e.kept else "—")
        tech = (e.technique or "?")[:60]
        lines.append(f"| {i} | {tech} | {before}→{after} | {delta} | {kept} | {reason} |")
    return "\n".join(lines)


def _load_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text()
    return (
        "You are a Metal kernel optimizer. Given the profile narrative + prior "
        "attempts + current .metal source, emit a new .metal source and a 2-3 "
        "sentence change summary in the JSON shape the user prompt requests."
    )


def _build_user_message(
    profile: ProfileResult,
    current_metal: str,
    mlx_reference: str,
    attempt_log_md: str,
    chip_aware: dict[str, Any] | None,
    retry_feedback: str | None = None,
) -> str:
    retry_block = (
        f"## Retry feedback\n\n{retry_feedback}\n\n" if retry_feedback else ""
    )
    return (
        "Write the next iteration of this Metal kernel.\n\n"
        + retry_block
        + f"## Profiler narrative\n\n{profile.narrative}\n\n"
        f"## Prior attempts on this kernel × chip\n\n{attempt_log_md}\n\n"
        f"## Chip-aware metrics (most recent run)\n\n"
        + (json.dumps(chip_aware, indent=2, default=str)[:3000] if chip_aware else "_(none — no .gputrace was attached)_")
        + "\n\n"
        f"## Current .metal source\n\n```metal\n{current_metal}\n```\n\n"
        f"## MLX reference (the spec — do NOT modify, just constraints)\n\n"
        f"```python\n{mlx_reference[:4000]}\n```\n\n"
        "## Output\n\n"
        "Strict JSON, no markdown fences:\n\n"
        '```json\n'
        '{\n'
        '  "new_metal_source": "// the full new .metal file contents — no truncation",\n'
        '  "change_summary": "2-3 sentences describing what you changed and why."\n'
        '}\n'
        '```\n'
    )


_JSON_BLOCK_RX = re.compile(r"\{.*\}\s*$", re.S)


def _parse_llm_json(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RX.search(s)
        if m:
            return json.loads(m.group(0))
        raise


def _accuracy_gate(
    kernel: str,
    staging_path: Path,
    active_metal_path: Path,
) -> tuple[bool, float | None, float | None, str]:
    """Swap staging into active path, run bench, decide kept-or-revert by
    correctness only. Restore active path on failure.

    Returns (passed, max_err, kernel_ms, note).
    """
    backup = active_metal_path.read_text()
    candidate = staging_path.read_text()
    active_metal_path.write_text(candidate)
    try:
        try:
            bench = run_bench(kernel)
        except Exception as e:
            active_metal_path.write_text(backup)
            return False, None, None, f"bench failed: {e}"
        if not bench.correct:
            active_metal_path.write_text(backup)
            return False, bench.max_err, bench.kernel_ms, "correctness_failed"
        return True, bench.max_err, bench.kernel_ms, "accuracy_passed"
    except Exception:
        active_metal_path.write_text(backup)
        raise


MAX_ACCURACY_RETRIES = 4


def optimize(
    profile: ProfileResult,
    *,
    provider: Provider,
    db: AttemptDB | None = None,
    chip_id: str | None = None,
) -> OptimizerResult:
    """LLM call → stage → accuracy gate. If accuracy fails, retry up to
    MAX_ACCURACY_RETRIES times. Every failed candidate is logged to AttemptDB
    with technique prefixed "Failed accuracy". Returns the first passing
    result, or the last failing result after the retry budget is exhausted.
    """
    db = db or AttemptDB()
    chip_id = chip_id or f"apple-{profile.chip_generation}"

    active_metal_path, set_name = _resolve_metal_path(profile.kernel, profile.chip_generation)
    current_metal = active_metal_path.read_text()
    mlx_path = _resolve_mlx_path(profile.kernel, set_name)
    mlx_reference = mlx_path.read_text() if mlx_path.is_file() else "(MLX reference not found)"

    last_result: OptimizerResult | None = None
    feedback: str | None = None

    for attempt_n in range(1, MAX_ACCURACY_RETRIES + 1):
        attempt_log = _render_attempt_log(db, profile.kernel, chip_id)

        resp = provider.generate(
            [
                Message("system", _load_system_prompt()),
                Message("user", _build_user_message(
                    profile, current_metal, mlx_reference, attempt_log,
                    profile.chip_aware_metrics,
                    retry_feedback=feedback,
                )),
            ],
            max_tokens=8000,
            temperature=0.3,
        )
        try:
            out = _parse_llm_json(resp.text)
        except Exception as e:
            last_result = OptimizerResult(
                kernel=profile.kernel, chip=chip_id,
                new_metal_source="", change_summary="",
                notes=f"LLM JSON parse failed (attempt {attempt_n}): {e}",
            )
            feedback = "Previous response was not valid JSON. Emit exactly the JSON shape requested."
            continue

        new_src = out.get("new_metal_source", "")
        summary = out.get("change_summary", "").strip()
        if not new_src.strip():
            last_result = OptimizerResult(
                kernel=profile.kernel, chip=chip_id,
                new_metal_source="", change_summary=summary,
                notes=f"empty new_metal_source (attempt {attempt_n})",
            )
            feedback = "Previous response had an empty new_metal_source. Emit the FULL kernel file."
            continue

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staging_path = STAGING_DIR / f"{profile.kernel}.metal"
        staging_path.write_text(new_src)

        passed, max_err, kernel_ms, note = _accuracy_gate(
            profile.kernel, staging_path, active_metal_path,
        )

        tech_label = summary[:200] or "(no summary)"
        if not passed:
            tech_label = f"Failed accuracy: {tech_label}"

        db.append(AttemptEntry(
            kernel=profile.kernel,
            chip=chip_id,
            technique=tech_label,
            diff="",
            files_touched=[str(active_metal_path.relative_to(REPO))],
            gputrace_metrics=profile.chip_aware_metrics or {},
            correctness_passed=passed,
            max_err=max_err,
            after_ms=kernel_ms,
            kept=False,
            rollback_reason="" if passed else f"accuracy_retry_{attempt_n}: {note}",
            notes=f"attempt {attempt_n}/{MAX_ACCURACY_RETRIES}; staged at {staging_path}; accuracy={note}",
        ))

        last_result = OptimizerResult(
            kernel=profile.kernel, chip=chip_id,
            new_metal_source=new_src,
            change_summary=summary,
            staging_path=staging_path,
            accuracy_passed=passed,
            accuracy_max_err=max_err,
            accuracy_kernel_ms=kernel_ms,
            notes=note,
        )
        if passed:
            return last_result

        feedback = (
            f"Your previous candidate failed the accuracy gate "
            f"(max_err={max_err if max_err is not None else 'unknown'}, "
            f"reason={note}). Retry with a different approach that preserves "
            f"the MLX reference's output to within rtol/atol = 1e-2."
        )

    assert last_result is not None
    return last_result


class OptimizerAgent:
    def __init__(self, provider: Provider, db: AttemptDB | None = None):
        self.provider = provider
        self.db = db or AttemptDB()

    def run(self, profile: ProfileResult, *, chip_id: str | None = None) -> OptimizerResult:
        return optimize(profile, provider=self.provider, db=self.db, chip_id=chip_id)
