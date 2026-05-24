"""Implementor agent — turn a ProfilerReport into a concrete .metal diff."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..profiler import ProfilerReport, SuggestedEdit
from ..providers import Message, Provider

REPO = Path(__file__).resolve().parents[2]
_PROMPT_PATH = REPO / "agent_steel" / "prompts" / "implementor.md"


# ---------------------------------------------------------------------------
# Output type.
# ---------------------------------------------------------------------------

@dataclass
class ImplementorResult:
    kernel: str
    technique_attempted: str        # the selected SuggestedEdit.technique
    diff: str                       # raw unified diff produced by the LLM
    applied_source: str | None      # what the .metal would look like after apply (None on parse fail)
    apply_succeeded: bool           # did the diff parse + apply cleanly?
    files_touched: list[str] = field(default_factory=list)
    notes: str = ""
    raw_llm_response: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Strategy selection — pick the top suggested edit that hasn't been tried.
# ---------------------------------------------------------------------------

def _pick_strategy(
    report: ProfilerReport,
    prior_attempts: list[str],
) -> SuggestedEdit | None:
    """Return the highest-ranked SuggestedEdit not present in prior_attempts.

    Match is case-insensitive substring on the `technique` field — the
    persistent-attempts log writes free-text descriptions, so a strict
    equality check would let near-duplicates through.
    """
    prior_lower = [p.lower() for p in prior_attempts]
    for edit in report.suggested_edits:
        t = edit.technique.lower()
        if any(t in p or p in t for p in prior_lower):
            continue
        return edit
    return None


# ---------------------------------------------------------------------------
# Locate the metal file path for the kernel (we need to put it in the prompt
# so the LLM emits diffs with correct `--- a/...` headers).
# ---------------------------------------------------------------------------

def _resolve_metal_path(kernel: str, set_name: str, chip: str) -> Path:
    """Return the active .metal file path for this (kernel, chip)."""
    chip_gen = (
        "m4" if "m4" in chip.lower()
        else "m3" if "m3" in chip.lower()
        else "m1" if "m1" in chip.lower()
        else "m2"
    )
    dir_path = REPO / "metal" / "kernels" / set_name / kernel
    flat = REPO / "metal" / "kernels" / set_name / f"{kernel}.metal"
    if dir_path.is_dir():
        # Prefer the chip-specific variant; else default.
        cand = dir_path / f"{chip_gen}.metal"
        if cand.is_file():
            return cand
        d = dir_path / "default.metal"
        if d.is_file():
            return d
        # Any single .metal in the dir as a last resort.
        for f in sorted(dir_path.iterdir()):
            if f.suffix == ".metal":
                return f
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"no .metal file for {kernel!r}")


# ---------------------------------------------------------------------------
# Diff application — string-level, no shelling out.
# ---------------------------------------------------------------------------

_FILE_HEADER = re.compile(r"^---\s+a/(\S+)\n\+\+\+\s+b/(\S+)\s*$", re.M)


def _parse_diff_files(diff: str) -> list[tuple[str, str]]:
    """Return list of (a_path, b_path) for each file header in the diff."""
    return [(m.group(1), m.group(2)) for m in _FILE_HEADER.finditer(diff)]


def _apply_diff_to_source(diff: str, source: str) -> tuple[str | None, str]:
    """Best-effort unified-diff applier.

    Returns (new_source, notes). Uses python's `unified_diff` semantics
    minimally — we only need to handle clean hunks since the LLM is
    instructed to produce minimal, well-formed diffs. For more robust
    application the Verifier should shell out to `patch -p1`.
    """
    lines = source.splitlines(keepends=True)
    new_lines: list[str] = []
    src_i = 0
    hunk_rx = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    in_hunk = False
    diff_lines = diff.splitlines(keepends=True)
    j = 0
    while j < len(diff_lines):
        L = diff_lines[j]
        if L.startswith("--- ") or L.startswith("+++ "):
            j += 1
            continue
        m = hunk_rx.match(L)
        if m:
            old_start = int(m.group(1))
            # copy unchanged lines from src_i up to old_start - 1
            while src_i < old_start - 1:
                if src_i >= len(lines):
                    return None, f"diff references line {old_start} past EOF"
                new_lines.append(lines[src_i])
                src_i += 1
            in_hunk = True
            j += 1
            continue
        if not in_hunk:
            j += 1
            continue
        if L.startswith(" "):
            # context line — must match
            if src_i >= len(lines) or lines[src_i].rstrip("\n") != L[1:].rstrip("\n"):
                return None, f"context mismatch at src line {src_i+1}"
            new_lines.append(lines[src_i])
            src_i += 1
        elif L.startswith("-"):
            # delete
            if src_i >= len(lines) or lines[src_i].rstrip("\n") != L[1:].rstrip("\n"):
                return None, f"delete mismatch at src line {src_i+1}"
            src_i += 1
        elif L.startswith("+"):
            # insert
            new_lines.append(L[1:])
        else:
            # blank or unknown — end of hunk
            in_hunk = False
        j += 1
    # tail
    while src_i < len(lines):
        new_lines.append(lines[src_i])
        src_i += 1
    return "".join(new_lines), "applied"


# ---------------------------------------------------------------------------
# Prompt assembly.
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text()
    return (
        "You are an Apple Metal kernel optimizer. Given a diagnostic + source + "
        "selected technique, output a unified diff and nothing else."
    )


def _build_user_message(
    report: ProfilerReport,
    selected: SuggestedEdit,
    metal_file_path: str,
    prior_attempts: list[str],
) -> str:
    packet = {
        "kernel": report.kernel,
        "selected_technique": {
            "technique": selected.technique,
            "rationale": selected.rationale,
            "target_lines": selected.target_lines,
            "expected_impact": selected.expected_impact,
        },
        "metal_file_path": metal_file_path,
        "metal_source": report.packet.get("metal_source", ""),
        "registry_entry": report.packet.get("registry_entry", ""),
        "constraints": {
            "pso_max_threads_per_tg": (
                report.packet.get("occupancy", {}).get("max_threads_per_tg")
            ),
            "tg_mem_bytes_in_use": (
                report.packet.get("occupancy", {}).get("tg_mem_bytes")
            ),
            "rtol": 1e-2,
            "atol": 1e-2,
            "chip": report.chip,
        },
        "prior_failed_techniques": prior_attempts,
    }
    return (
        "Produce a unified diff for the selected_technique applied to "
        "metal_source. Output only the diff.\n\n"
        + json.dumps(packet, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def implement(
    report: ProfilerReport,
    *,
    provider: Provider,
    prior_attempts: list[str] | None = None,
    set_name: str | None = None,
) -> ImplementorResult:
    """Generate a diff for the top untried suggested edit on `report.kernel`."""
    prior = prior_attempts or []
    selected = _pick_strategy(report, prior)
    if selected is None:
        return ImplementorResult(
            kernel=report.kernel,
            technique_attempted="",
            diff="",
            applied_source=None,
            apply_succeeded=False,
            notes=(
                "No untried suggested edit remains. Either the profiler returned "
                "no suggestions, or every suggestion is in prior_attempts."
            ),
        )

    # Locate the metal file we'll be diffing against.
    if set_name is None:
        set_name = report.packet.get("set", "common")
    try:
        metal_path = _resolve_metal_path(report.kernel, set_name, report.chip)
    except FileNotFoundError as e:
        return ImplementorResult(
            kernel=report.kernel,
            technique_attempted=selected.technique,
            diff="",
            applied_source=None,
            apply_succeeded=False,
            notes=str(e),
        )
    metal_rel = str(metal_path.relative_to(REPO))

    # Ask the LLM.
    resp = provider.generate(
        [
            Message("system", _load_system_prompt()),
            Message("user", _build_user_message(report, selected, metal_rel, prior)),
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    raw = resp.text.strip()

    # Strip code fences if the model added them despite instructions.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:diff|patch)?\n", "", raw)
        raw = re.sub(r"\n```\s*$", "", raw)

    files_touched = [b for _, b in _parse_diff_files(raw)]

    # Apply against the live source on disk to produce the would-be-applied source.
    current_source = metal_path.read_text()
    new_source, apply_note = _apply_diff_to_source(raw, current_source)
    apply_ok = new_source is not None

    return ImplementorResult(
        kernel=report.kernel,
        technique_attempted=selected.technique,
        diff=raw,
        applied_source=new_source,
        apply_succeeded=apply_ok,
        files_touched=files_touched,
        notes=apply_note,
        raw_llm_response=resp.text,
    )


class ImplementorAgent:
    """Thin OO wrapper for symmetry with future agents (Verifier, Loop)."""

    def __init__(self, provider: Provider):
        self.provider = provider

    def run(
        self,
        report: ProfilerReport,
        prior_attempts: list[str] | None = None,
        set_name: str | None = None,
    ) -> ImplementorResult:
        return implement(
            report,
            provider=self.provider,
            prior_attempts=prior_attempts,
            set_name=set_name,
        )
