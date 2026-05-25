"""Welder — authoring + polish agent.

Lives outside the Profiler→Optimizer→Verifier closed loop. Invoked twice
per new-kernel session: once to create the MLX/registry/Metal files, once
after the loop finishes to polish for PR.
"""
from __future__ import annotations
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..providers import Message, Provider

REPO = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).parent / "prompts"
MAX_CREATE_RETRIES = 4


@dataclass
class CreateResult:
    kernel: str
    set_name: str
    accuracy_passed: bool
    metal_path: Path | None = None
    mlx_path: Path | None = None
    files_written: list[Path] = field(default_factory=list)
    notes: str = ""
    design_notes: str = ""


@dataclass
class PolishResult:
    kernel: str
    ready_for_pr: bool
    issues: list[str] = field(default_factory=list)
    suggested_pr_title: str = ""
    suggested_pr_body: str = ""
    cleanup_commands: list[str] = field(default_factory=list)


_JSON_RX = re.compile(r"\{.*\}\s*$", re.S)


def _parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_RX.search(s)
        if m:
            return json.loads(m.group(0))
        raise


def _read_sample_registry(set_name: str) -> str:
    p = REPO / "mlx" / "kernels" / set_name / "registry.py"
    return p.read_text() if p.is_file() else ""


def _read_sample_mlx(set_name: str) -> str:
    d = REPO / "mlx" / "kernels" / set_name
    if not d.is_dir():
        return ""
    for f in sorted(d.iterdir()):
        if f.suffix == ".py" and f.name != "registry.py":
            return f"### example: {f.name}\n\n```python\n{f.read_text()}\n```"
    return ""


def _sniff_chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _bench_correctness(kernel: str) -> tuple[bool, float | None, str]:
    """Return (correct, max_err, raw_output)."""
    try:
        r = subprocess.run(
            ["./bench", kernel, "--no-save", "--", "--iters", "5", "--warmup", "2"],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        )
    except Exception as e:
        return False, None, f"bench invocation failed: {e}"
    out = r.stdout + r.stderr
    m = re.search(r"correctness\s+:\s*(✓|✗).*?max_err=([\d.eE+-]+)", out)
    if not m:
        return False, None, out[-2000:]
    return m.group(1) == "✓", float(m.group(2)), out[-1000:]


def _run_external_reference(reference_code: str, mlx_path: Path) -> tuple[bool, str]:
    """Eval user-supplied PyTorch/NumPy reference and the MLX module on the
    same input; compare within 1e-2. `reference_code` must define `ref(*args)`.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mlx_ref", mlx_path)
    if spec is None or spec.loader is None:
        return False, "could not load MLX module"
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return False, f"MLX module raised on import: {e}"
    if not hasattr(mod, "Model"):
        return False, "MLX module has no `Model` class"

    ns: dict = {}
    try:
        exec(reference_code, ns)
    except Exception as e:
        return False, f"reference code raised on exec: {e}"
    if "ref" not in ns:
        return False, "reference code must define `ref(*inputs)`"

    try:
        import mlx.core as mx
        x = mx.random.uniform(shape=(16, 16384))
        ours = mod.Model().forward(x)
        theirs = ns["ref"](x)
        diff = float(mx.max(mx.abs(ours - theirs)).item())
        return diff <= 1e-2, f"max_abs_diff={diff:.4e}"
    except Exception as e:
        return False, f"reference comparison raised: {e}"


def _resolve_paths(kernel: str, set_name: str) -> tuple[Path, Path, Path, Path]:
    """Return (mlx_path, registry_path, metal_path, kernels_md)."""
    return (
        REPO / "mlx" / "kernels" / set_name / f"{kernel}.py",
        REPO / "mlx" / "kernels" / set_name / "registry.py",
        REPO / "metal" / "kernels" / set_name / f"{kernel}.metal",
        REPO / "KERNELS.md",
    )


def _append_registry_entry(registry_path: Path, kernel: str, entry_expr: str) -> None:
    """Append `<expr>` to the bottom of registry.py if it isn't already there.

    The LLM emits either a `REGISTRY[name] = dict(...)` line or a helper call
    like `ew("name", "metal_func", ...)`. Both are appended verbatim.
    """
    src = registry_path.read_text()
    snippet = entry_expr.strip()
    if kernel in src:
        return  # already registered
    if not src.endswith("\n"):
        src += "\n"
    registry_path.write_text(src + "\n" + snippet + "\n")


def create(
    kernel: str,
    *,
    provider: Provider,
    description: str,
    reference_code: str | None = None,
    set_hint: str = "common",
) -> CreateResult:
    """Author MLX + registry + Metal for a brand-new kernel. Returns when
    accuracy passes both stages OR after MAX_CREATE_RETRIES attempts.
    """
    chip = _sniff_chip()
    feedback: str | None = None
    last_result: CreateResult | None = None

    for attempt in range(1, MAX_CREATE_RETRIES + 1):
        sample_registry = _read_sample_registry(set_hint)
        sample_mlx = _read_sample_mlx(set_hint)

        retry_block = f"## Retry feedback\n\n{feedback}\n\n" if feedback else ""
        user_msg = (
            f"Author a new kernel for MetalBench.\n\n"
            f"## Kernel\n\nname: `{kernel}`\nchip detected: `{chip}`\nsuggested set: `{set_hint}`\n\n"
            f"## Description\n\n{description}\n\n"
            + (f"## External reference (for Stage B accuracy)\n\n```python\n{reference_code}\n```\n\n"
               if reference_code else "")
            + retry_block
            + f"## Sample MLX file from `{set_hint}/`\n\n{sample_mlx}\n\n"
            f"## Current `mlx/kernels/{set_hint}/registry.py`\n\n```python\n{sample_registry[:4000]}\n```\n"
        )

        resp = provider.generate(
            [
                Message("system", (PROMPTS / "welder_create.md").read_text()),
                Message("user", user_msg),
            ],
            max_tokens=8000, temperature=0.2,
        )
        try:
            out = _parse_json(resp.text)
        except Exception as e:
            feedback = f"Previous response was not valid JSON: {e}"
            last_result = CreateResult(kernel=kernel, set_name=set_hint,
                                       accuracy_passed=False,
                                       notes=f"json parse failed (attempt {attempt})")
            continue

        set_name = out.get("set") or set_hint
        mlx_text = out.get("mlx_reference", "")
        registry_expr = out.get("registry_entry", "")
        metal_text = out.get("metal_source", "")
        km_row = out.get("kernels_md_row", "")
        design = (out.get("design_notes") or "").strip()

        if not (mlx_text.strip() and metal_text.strip() and registry_expr.strip()):
            feedback = "Empty mlx_reference, registry_entry, or metal_source. Emit all three full files."
            last_result = CreateResult(kernel=kernel, set_name=set_name,
                                       accuracy_passed=False,
                                       notes=f"empty fields (attempt {attempt})")
            continue

        mlx_path, reg_path, metal_path, km_path = _resolve_paths(kernel, set_name)
        mlx_path.parent.mkdir(parents=True, exist_ok=True)
        metal_path.parent.mkdir(parents=True, exist_ok=True)
        mlx_path.write_text(mlx_text)
        metal_path.write_text(metal_text)
        _append_registry_entry(reg_path, kernel, registry_expr)
        if km_row and km_row.strip() and km_path.is_file():
            km_text = km_path.read_text()
            if kernel not in km_text:
                km_path.write_text(km_text.rstrip() + "\n" + km_row.strip() + "\n")

        files = [mlx_path, reg_path, metal_path]
        if km_path.is_file():
            files.append(km_path)

        # Stage A: ./bench correctness
        correct, max_err, raw = _bench_correctness(kernel)
        if not correct:
            feedback = (
                f"Stage A failed: ./bench {kernel} did not report correct. "
                f"max_err={max_err}. Tail of bench output:\n\n{raw[-1500:]}"
            )
            last_result = CreateResult(kernel=kernel, set_name=set_name,
                                       accuracy_passed=False,
                                       metal_path=metal_path, mlx_path=mlx_path,
                                       files_written=files, design_notes=design,
                                       notes=f"stage_A_failed (attempt {attempt})")
            continue

        # Stage B: external reference (only when provided)
        if reference_code:
            ok_b, note_b = _run_external_reference(reference_code, mlx_path)
            if not ok_b:
                feedback = f"Stage B failed (MLX vs external reference): {note_b}"
                last_result = CreateResult(kernel=kernel, set_name=set_name,
                                           accuracy_passed=False,
                                           metal_path=metal_path, mlx_path=mlx_path,
                                           files_written=files, design_notes=design,
                                           notes=f"stage_B_failed (attempt {attempt}): {note_b}")
                continue

        return CreateResult(
            kernel=kernel, set_name=set_name, accuracy_passed=True,
            metal_path=metal_path, mlx_path=mlx_path, files_written=files,
            design_notes=design,
            notes=f"created at attempt {attempt}; max_err={max_err}",
        )

    assert last_result is not None
    return last_result


def polish(
    kernel: str,
    *,
    provider: Provider,
    chip: str,
    loop_result: dict[str, Any],
) -> PolishResult:
    """Post-loop PR readiness check. Calls the LLM with git state + the
    loop's summary; returns a structured ready-or-fix decision."""
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO,
        capture_output=True, text=True, timeout=10,
    ).stdout
    session_diff = subprocess.run(
        ["git", "diff", "session.json"], cwd=REPO,
        capture_output=True, text=True, timeout=10,
    ).stdout[:4000]
    best_diff = subprocess.run(
        ["git", "diff", "best_times.md"], cwd=REPO,
        capture_output=True, text=True, timeout=10,
    ).stdout[:2000]

    payload = {
        "kernel": kernel, "chip": chip, "loop_result": loop_result,
        "git_status": git_status,
        "session_json_diff": session_diff,
        "best_times_diff": best_diff,
    }
    resp = provider.generate(
        [
            Message("system", (PROMPTS / "welder_polish.md").read_text()),
            Message("user", "Polish this session for PR.\n\n" + json.dumps(payload, indent=2)),
        ],
        max_tokens=2000, temperature=0.1,
    )
    try:
        out = _parse_json(resp.text)
    except Exception as e:
        return PolishResult(kernel=kernel, ready_for_pr=False,
                            issues=[f"polish-LLM returned non-JSON: {e}"])
    return PolishResult(
        kernel=kernel,
        ready_for_pr=bool(out.get("ready_for_pr")),
        issues=list(out.get("issues") or []),
        suggested_pr_title=out.get("suggested_pr_title", ""),
        suggested_pr_body=out.get("suggested_pr_body", ""),
        cleanup_commands=list(out.get("cleanup_commands") or []),
    )


class WelderAgent:
    """OO wrapper for symmetry with the other agents."""

    def __init__(self, provider: Provider):
        self.provider = provider

    def create(self, kernel: str, *, description: str,
               reference_code: str | None = None, set_hint: str = "common") -> CreateResult:
        return create(kernel, provider=self.provider, description=description,
                      reference_code=reference_code, set_hint=set_hint)

    def polish(self, kernel: str, *, chip: str, loop_result: dict[str, Any]) -> PolishResult:
        return polish(kernel, provider=self.provider, chip=chip, loop_result=loop_result)
