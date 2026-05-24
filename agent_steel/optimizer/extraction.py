"""Extraction — pull candidate optimization techniques from multiple sources.

Sources (queried in order of confidence):
1. patterns.json hand-curated wins (high confidence, exact-match)
2. profiler.suggested_edits (LLM-driven, kernel-specific)
3. history DB — what's been kept on this kernel × chip before
4. anti-patterns filter (negative — drops candidates that match known failures)

Returns a deduped, ranked list of Candidates the Implementor can act on.
Pure Python; no LLM call here. Deterministic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ..history import AttemptDB
from ..profiler import ProfilerReport
from . import knowledge_base as kb


@dataclass
class Candidate:
    technique: str                                 # short label
    rationale: str                                 # 1-2 sentences
    source: str                                    # "patterns" | "profiler_llm" | "history"
    confidence: float = 0.5                        # 0..1
    target_lines: str = ""                         # optional code anchor
    expected_impact: str = ""                      # human estimate
    pattern_id: str = ""                           # patterns.json id (if from kb)
    wins_on: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel-kind classification — drives pattern lookup.
# ---------------------------------------------------------------------------

def _classify_kernel_kinds(report: ProfilerReport) -> list[str]:
    """Heuristic mapping from packet → ordered list of patterns.json kind buckets.

    Returns a list so kernels that span multiple categories (full-set fused
    networks contain conv + matmul + fused patterns) can surface techniques
    from all relevant buckets, not just one.
    """
    src = report.packet.get("source_analysis") or {}
    name = report.kernel.lower()
    kinds: list[str] = []

    # Full-set kernels contain multiple op types — query all plausible buckets.
    if name in ("alexnet", "resnet", "mlp", "densenet", "transformer_block",
                "llama_decoder_layer"):
        kinds.append("fused")
        # CNN-shaped Full kernels — try conv patterns too
        if name in ("alexnet", "resnet", "densenet"):
            kinds.append("conv")
        # Matmul-heavy Full kernels — try matmul patterns
        if name in ("mlp", "transformer_block", "llama_decoder_layer"):
            kinds.append("matmul")
        return kinds

    if "conv" in name or "depthwise" in name:
        return ["conv"]
    if any(s in name for s in ("matmul", "mm", "matvec", "outer_product",
                                "linear", "gemm")):
        return ["matmul"]
    if any(s in name for s in ("relu", "gelu", "sigmoid", "tanh", "swish",
                                "softplus", "hardsigmoid", "elu", "selu",
                                "abs", "exp", "log", "rsqrt", "clip", "mish",
                                "softsign", "hardswish")):
        return ["elementwise"]
    if any(s in name for s in ("softmax", "norm", "argmax", "sum", "mean",
                                "variance", "cumsum", "cumprod", "logsumexp",
                                "dot_product", "manhattan", "cosine",
                                "frobenius", "l1_norm", "l2_norm")):
        return ["reduction"]
    if src.get("has_simdgroup_matrix") or src.get("loop_count", 0) >= 3:
        return ["matmul"]
    return ["elementwise"]


def _classify_kernel_kind(report: ProfilerReport) -> str:
    """Back-compat shim — returns the primary (first) kind."""
    return _classify_kernel_kinds(report)[0]


# ---------------------------------------------------------------------------
# Applies-when constraint matcher.
# ---------------------------------------------------------------------------

def _matches_applies_when(
    constraints: dict[str, Any],
    report: ProfilerReport,
    kernel_kind: str,
) -> bool:
    """Evaluate the pattern's `applies_when` dict against the packet."""
    if not constraints:
        return True

    pkt = report.packet
    src = pkt.get("source_analysis") or {}
    roof = pkt.get("roofline") or {}
    disp = pkt.get("dispatch_check") or {}

    for key, expected in constraints.items():
        if key == "loop_count_min":
            if (src.get("loop_count") or 0) < expected:
                return False
        elif key == "has_threadgroup_mem":
            if bool(src.get("has_threadgroup_mem")) != bool(expected):
                return False
        elif key == "has_simdgroup_matrix":
            if bool(src.get("has_simdgroup_matrix")) != bool(expected):
                return False
        elif key == "has_unroll_pragma":
            if bool(src.get("has_unroll_pragma")) != bool(expected):
                return False
        elif key == "has_float4":
            if bool(src.get("has_float4")) != bool(expected):
                return False
        elif key == "has_simd_reduction":
            if bool(src.get("has_simd_reduction")) != bool(expected):
                return False
        elif key == "kernel_kind_one_of":
            if kernel_kind not in expected:
                return False
        elif key == "threadgroup_exceeds_pso_limit":
            if bool(disp.get("threadgroup_within_pso_limit") is False) != bool(expected):
                return False
        elif key == "channel_count_max":
            # Best-effort — we don't reliably know channel count. Pass-through unless
            # the profiler packet later surfaces it explicitly.
            continue
        elif key in ("k_dim_min", "m_dim_min", "n_dim_min", "n_inputs_min"):
            # Dimension constraints — pass-through for v1 (we don't reliably
            # parse these out of registry shapes yet).
            continue
        else:
            # Unknown constraint key → conservative: don't drop the candidate.
            continue

    return True


# ---------------------------------------------------------------------------
# Candidate-from-pattern constructor with confidence weighting.
# ---------------------------------------------------------------------------

def _candidate_from_pattern(p: dict[str, Any]) -> Candidate:
    n_wins = len(p.get("wins_on") or [])
    # confidence: 0.6 base + 0.05 per prior win (capped at 0.95)
    confidence = min(0.95, 0.6 + 0.05 * n_wins)
    return Candidate(
        technique=p.get("technique", ""),
        rationale=p.get("rationale", ""),
        source="patterns",
        confidence=confidence,
        pattern_id=p.get("id", ""),
        wins_on=list(p.get("wins_on") or []),
    )


def _candidate_from_profiler_edit(edit, *, base_confidence: float = 0.5) -> Candidate:
    return Candidate(
        technique=edit.technique,
        rationale=edit.rationale,
        source="profiler_llm",
        confidence=base_confidence,
        target_lines=edit.target_lines,
        expected_impact=edit.expected_impact,
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def extract(
    report: ProfilerReport,
    *,
    chip: str | None = None,
    db: AttemptDB | None = None,
) -> list[Candidate]:
    """Return a ranked, deduped list of Candidate techniques for this kernel.

    The caller (loop) picks the top untried Candidate and hands it to the
    Implementor.
    """
    chip = chip or report.chip
    db = db or AttemptDB()

    bottleneck = report.bottleneck_class or "compute-bound"
    # Strip the "(latency-dominated ...)" suffix the roofline adds.
    bottleneck_root = bottleneck.split(" (")[0]
    kernel_kinds = _classify_kernel_kinds(report)
    kernel_kind = kernel_kinds[0]  # primary, for applies_when checks

    candidates: list[Candidate] = []

    # ----- 1. From patterns.json — query every plausible kind bucket -----
    # Pass the kind currently being queried to applies_when, not just the primary,
    # so patterns under "conv" don't reject a kernel classified as ["fused", "conv"].
    for kind in kernel_kinds:
        for p in kb.lookup(bottleneck_root, kind):
            if not _matches_applies_when(p.get("applies_when") or {}, report, kind):
                continue
            candidates.append(_candidate_from_pattern(p))

    # The sanity-override class lives under `under_roofline_likely_latency_or_stall`
    # — patterns may file under "any" there.
    if not candidates and bottleneck_root.startswith("under_roofline"):
        for p in kb.lookup(bottleneck_root, "any"):
            candidates.append(_candidate_from_pattern(p))

    # ----- 2. From profiler.suggested_edits -----
    for e in report.suggested_edits:
        candidates.append(_candidate_from_profiler_edit(e))

    # ----- 3. From history DB — kept attempts on similar kernels -----
    # For now we only look at this kernel's own history (cross-kernel
    # transfer lands when wins.jsonl is added).
    for attempt in db.read(report.kernel, chip):
        if attempt.kept and attempt.technique:
            candidates.append(Candidate(
                technique=attempt.technique,
                rationale=f"This exact technique kept a {attempt.improvement_pct or 0:.1f}% improvement on this kernel before. Re-applying may stack additional gains.",
                source="history",
                confidence=0.7,
            ))

    # ----- 4. Negative filter — drop prior failures (case-insensitive substring) -----
    prior_failed = [
        (t.lower(), reason)
        for (t, reason) in db.failed_techniques(report.kernel, chip)
    ]
    filtered: list[Candidate] = []
    for c in candidates:
        tlow = c.technique.lower()
        if any(t in tlow or tlow in t for (t, _) in prior_failed):
            continue
        filtered.append(c)

    # ----- 5. Dedupe by technique (case-insensitive substring) -----
    deduped: list[Candidate] = []
    seen: list[str] = []
    for c in filtered:
        tlow = c.technique.lower()
        if any(s in tlow or tlow in s for s in seen):
            continue
        deduped.append(c)
        seen.append(tlow)

    # ----- 6. Rank: confidence desc, then patterns first, then profiler_llm -----
    source_rank = {"patterns": 0, "history": 1, "profiler_llm": 2}
    deduped.sort(key=lambda c: (-c.confidence, source_rank.get(c.source, 3)))

    return deduped
