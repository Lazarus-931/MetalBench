"""Agent Steel — CLI entry point.

Usage:

    # Optimize an existing kernel (profiler-only for now; later: full pipeline)
    python -m agent_steel --kernel-name softmax_attention
    python -m agent_steel --kernel-name relu --no-llm        # deterministic only

    # Author a new kernel from scratch (Profiler→Designer→Implementor→Verifier loop, WIP)
    python -m agent_steel --new <kernel_name>

    # Run N workers in parallel over multiple kernels
    python -m agent_steel --kernel-name softmax,relu,gelu --parallel 3

Provider selection — three flavours, all routed through one ProviderConfig:

    # OpenAI hosted (default)
    OPENAI_API_KEY=...  python -m agent_steel --kernel-name relu

    # Anthropic Claude
    ANTHROPIC_API_KEY=...  python -m agent_steel --provider anthropic --kernel-name relu

    # OpenRouter — OpenAI-compatible aggregator (one key, hundreds of models)
    OPENAI_API_KEY=<openrouter-key>  python -m agent_steel \\
        --provider openrouter --model meta-llama/llama-3.1-70b-instruct \\
        --kernel-name relu

    # Ollama — local OpenAI-compatible server, no key needed
    python -m agent_steel --provider ollama --model llama3.1:70b --kernel-name relu

The agent respects MetalBench's PR-based workflow: it writes proposed edits
into the working tree, runs `./bench` to verify, but does NOT commit, push,
or modify anything outside the kernel's `.metal` file (and `registry.py` if
a dispatch-shape change is justified).
"""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .providers import Provider, get_provider
from .profiler import profile


import os as _os

_PROVIDER_PRESETS = {
    "openai":     {"backend": "openai",    "base_url": None, "model": "gpt-4o-mini",
                   "api_key_env": "OPENAI_API_KEY"},
    "anthropic":  {"backend": "anthropic", "base_url": None, "model": "claude-sonnet-4-6",
                   "api_key_env": "ANTHROPIC_API_KEY"},
    "deepseek":   {"backend": "openai",    "base_url": "https://api.deepseek.com/v1",
                   "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "openrouter": {"backend": "openai",    "base_url": "https://openrouter.ai/api/v1",
                   "model": "anthropic/claude-sonnet-4",
                   "api_key_env": "OPENROUTER_API_KEY"},
    "ollama":     {"backend": "openai",    "base_url": "http://localhost:11434/v1",
                   "model": "llama3.1:70b", "api_key_env": None},
}


def _make_provider(name: str, model_override: str | None) -> Provider:
    if name not in _PROVIDER_PRESETS:
        raise SystemExit(
            f"unknown --provider {name!r}. Choose one of: {list(_PROVIDER_PRESETS)}"
        )
    p = _PROVIDER_PRESETS[name]
    key_env = p.get("api_key_env")
    api_key = _os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise SystemExit(f"--provider {name!r} requires {key_env} in the environment.")
    return get_provider(
        p["backend"],
        api_key=api_key,
        base_url=p["base_url"],
        default_model=model_override or p["model"],
    )


def _print_report(r, output: str) -> None:
    if output == "json":
        print(json.dumps({
            "kernel": r.kernel, "chip": r.chip,
            "generation": r.chip_generation, "variant": r.chip_variant,
            "narrative": r.narrative,
        }, indent=2))
        return
    print(f"\n  kernel       : {r.kernel}  ({r.chip})")
    print(f"  generation   : {r.chip_generation} ({r.chip_variant})")
    print(f"  kernel_ms    : {r.bench.kernel_ms}")
    print(f"  GFLOPS / BW  : {r.bench.gflops} / {r.bench.gbps} GB/s")
    print(f"\n  narrative:\n\n{r.narrative}\n")


def _run_one(kernel: str, args):
    """Run the configured stages for one kernel. Returns (kernel, ProfileResult | LoopResult | Exception)."""
    try:
        if args.no_llm:
            raise SystemExit("--no-llm is no longer supported (Profiler is LLM-only). Pass --provider <provider>.")
        prov = _make_provider(args.provider, args.model)
        if args.loop:
            from .loop import run_loop, GreedyStrategy
            strategy = GreedyStrategy(
                max_rounds=args.max_rounds,
                max_no_improvement=args.max_no_improvement,
            )
            return kernel, run_loop(kernel, provider=prov, strategy=strategy)
        report = profile(kernel, provider=prov, gputrace_path=args.gputrace)
        return kernel, report
    except Exception as e:
        return kernel, e


def _print_loop_result(r) -> None:
    print(f"\n  kernel       : {r.kernel}  ({r.chip})")
    print(f"  rounds       : {r.rounds_run}  (kept: {r.kept_attempts})")
    print(f"  initial_ms   : {r.initial_ms}")
    print(f"  best_ms      : {r.best_ms}")
    if r.overall_improvement_pct is not None:
        print(f"  improvement  : {r.overall_improvement_pct:.1f}%")
    print(f"  terminated   : {r.termination_reason}")
    if r.attempts:
        print(f"\n  attempts (most recent):")
        for a in r.attempts[-min(5, len(r.attempts)):]:
            status = "KEPT" if a.kept else f"rev:{a.rollback_reason}"
            imp = f"{a.improvement_pct:+.1f}%" if a.improvement_pct is not None else "—"
            print(f"    [{status:>15}] {imp:>8}  {a.technique[:60]}")


def _split_kernels(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _run_welder(args) -> int:
    """Welder flow: create one or more new kernels, optionally chain into --loop,
    then call polish on each. Returns process exit code."""
    from .welder import create as welder_create, polish as welder_polish

    kernels = _split_kernels(args.welder_kernel or "")
    if not kernels:
        print("--welder takes at least one kernel name.", file=sys.stderr)
        return 2
    if not args.description:
        print("--welder requires --description '<what the kernel does>'.", file=sys.stderr)
        return 2

    reference_code: str | None = None
    if args.reference:
        rp = Path(args.reference)
        if not rp.is_file():
            print(f"--reference {args.reference}: file not found", file=sys.stderr)
            return 2
        reference_code = rp.read_text()

    prov = _make_provider(args.provider, args.model)
    failures = 0

    for k in kernels:
        print(f"\n[welder/create] {k}")
        cr = welder_create(
            k, provider=prov,
            description=args.description,
            reference_code=reference_code,
            set_hint=args.set_hint,
        )
        if not cr.accuracy_passed:
            print(f"[welder/create] {k}: FAILED — {cr.notes}", file=sys.stderr)
            failures += 1
            continue
        print(f"[welder/create] {k}: ✓ accuracy passed ({cr.notes})")
        print(f"  design: {cr.design_notes}")
        print(f"  files:  {', '.join(str(p.relative_to(Path.cwd())) for p in cr.files_written)}")

        if args.loop:
            from .loop import run_loop, GreedyStrategy
            strategy = GreedyStrategy(
                max_rounds=args.max_rounds,
                max_no_improvement=args.max_no_improvement,
            )
            print(f"[welder→loop] starting perf loop on {k}")
            lr = run_loop(k, provider=prov, strategy=strategy)
            _print_loop_result(lr)
            loop_dict = {
                "initial_ms": lr.initial_ms, "best_ms": lr.best_ms,
                "overall_improvement_pct": lr.overall_improvement_pct,
                "rounds_run": lr.rounds_run, "kept_attempts": lr.kept_attempts,
                "termination_reason": lr.termination_reason,
            }
        else:
            loop_dict = {"note": "loop skipped (no --loop)"}

        print(f"\n[welder/polish] {k}")
        pr = welder_polish(k, provider=prov, chip=f"apple-detected", loop_result=loop_dict)
        print(f"  ready_for_pr: {pr.ready_for_pr}")
        if pr.issues:
            print("  issues:")
            for i in pr.issues:
                print(f"    - {i}")
        if pr.cleanup_commands:
            print("  suggested cleanup commands (review before running):")
            for c in pr.cleanup_commands:
                print(f"    $ {c}")
        if pr.suggested_pr_title:
            print(f"\n  PR title: {pr.suggested_pr_title}")
            print(f"  PR body:\n{pr.suggested_pr_body}")

    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_steel",
        description="Agent Steel — agentic kernel-authoring harness for MetalBench.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--kernel-name", dest="kernel_name",
        help="Optimize one or more existing kernels (comma-separated).",
    )
    mode.add_argument(
        "--welder", dest="welder_kernel",
        help="Author one or more NEW kernels (comma-separated) via the Welder agent. "
             "Writes MLX + registry + Metal files; chains into --loop afterwards if also passed.",
    )

    ap.add_argument("--description", default=None,
                    help="(--welder only) Natural-language description of what the kernel does.")
    ap.add_argument("--reference", default=None,
                    help="(--welder only) Path to a Python file defining `ref(*inputs)` used as "
                         "the external accuracy oracle (Stage B). Recommended.")
    ap.add_argument("--set", dest="set_hint", default="common",
                    choices=["common", "standard", "full"],
                    help="(--welder only) Which kernel set to add the new kernel to.")

    ap.add_argument(
        "--provider", default="anthropic",
        choices=list(_PROVIDER_PRESETS),
        help="LLM backend (default: anthropic). 'ollama' and 'openrouter' both "
             "use the OpenAI-compatible client with a different base_url.",
    )
    ap.add_argument("--model", default=None, help="Override the provider's default model.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM calls — use deterministic-only roofline output.")

    ap.add_argument("--gputrace", default=None,
                    help="Path to a .gputrace bundle to enrich the diagnostic.")

    ap.add_argument("--loop", action="store_true",
                    help="Run the full agent loop (profile → optimize → verify → repeat). "
                         "Without this flag, only the Profiler runs.")
    ap.add_argument("--max-rounds", type=int, default=5,
                    help="Max loop rounds before termination (default 5).")
    ap.add_argument("--max-no-improvement", type=int, default=3,
                    help="Stop after this many consecutive rounds without a kept improvement.")

    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of worker threads when --kernel-name lists multiple.")

    ap.add_argument("--output", choices=["json", "text"], default="text")

    args = ap.parse_args(argv)

    if args.welder_kernel:
        return _run_welder(args)

    kernels = _split_kernels(args.kernel_name or "")
    if not kernels:
        print("--kernel-name takes at least one kernel name.", file=sys.stderr)
        return 2

    if args.parallel <= 1 or len(kernels) == 1:
        for k in kernels:
            name, result = _run_one(k, args)
            if isinstance(result, Exception):
                print(f"\n[{name}] ERROR: {result}", file=sys.stderr)
                continue
            if args.loop:
                _print_loop_result(result)
            else:
                _print_report(result, args.output)
        return 0

    workers = min(args.parallel, len(kernels))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_one, k, args): k for k in kernels}
        for fut in as_completed(futs):
            name, result = fut.result()
            if isinstance(result, Exception):
                print(f"\n[{name}] ERROR: {result}", file=sys.stderr)
                continue
            if args.loop:
                _print_loop_result(result)
            else:
                _print_report(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
