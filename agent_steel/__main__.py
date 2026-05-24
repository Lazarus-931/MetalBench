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
from dataclasses import asdict

from .providers import Provider, get_provider
from .profiler import profile, ProfilerReport


# Provider name → (factory short-name, default base_url, default model)
_PROVIDER_PRESETS = {
    "openai":     {"backend": "openai",    "base_url": None, "model": "gpt-4o-mini"},
    "anthropic":  {"backend": "anthropic", "base_url": None, "model": "claude-sonnet-4-6"},
    "openrouter": {"backend": "openai",    "base_url": "https://openrouter.ai/api/v1",
                   "model": "anthropic/claude-sonnet-4"},
    "ollama":     {"backend": "openai",    "base_url": "http://localhost:11434/v1",
                   "model": "llama3.1:70b"},
}


def _make_provider(name: str, model_override: str | None) -> Provider:
    if name not in _PROVIDER_PRESETS:
        raise SystemExit(
            f"unknown --provider {name!r}. Choose one of: {list(_PROVIDER_PRESETS)}"
        )
    p = _PROVIDER_PRESETS[name]
    return get_provider(
        p["backend"],
        base_url=p["base_url"],
        default_model=model_override or p["model"],
    )


def _print_report(r: ProfilerReport, output: str) -> None:
    if output == "json":
        print(json.dumps({
            "kernel": r.kernel, "chip": r.chip,
            "bottleneck_class": r.bottleneck_class,
            "sol": r.sol, "confidence": r.confidence,
            "code_analysis": r.code_analysis,
            "suggested_edits": [asdict(e) for e in r.suggested_edits],
        }, indent=2))
        return
    print(f"\n  kernel       : {r.kernel}  ({r.chip})")
    print(f"  bottleneck   : {r.bottleneck_class}")
    print(f"  sol          : {r.sol*100:.0f}%")
    print(f"  confidence   : {r.confidence:.2f}")
    print(f"\n  analysis     : {r.code_analysis}\n")
    if r.suggested_edits:
        print("  suggested edits (ranked):")
        for i, e in enumerate(r.suggested_edits, 1):
            print(f"    {i}. {e.technique}")
            print(f"       why: {e.rationale}")
            print(f"       where: {e.target_lines}")
            print(f"       impact: {e.expected_impact}\n")
    else:
        print("  no edits suggested\n")


def _run_one(kernel: str, args):
    """Run the configured stages for one kernel. Returns (kernel, ProfilerReport | LoopResult | Exception)."""
    try:
        prov = None if args.no_llm else _make_provider(args.provider, args.model)
        if args.loop:
            from .loop import run_loop, GreedyStrategy
            if prov is None:
                raise RuntimeError("--loop requires a provider (cannot run with --no-llm)")
            strategy = GreedyStrategy(
                max_rounds=args.max_rounds,
                max_no_improvement=args.max_no_improvement,
            )
            return kernel, run_loop(kernel, provider=prov, strategy=strategy)
        report = profile(kernel, provider=prov, skip_llm=args.no_llm,
                         gputrace_path=args.gputrace)
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
        kept = [a for a in r.attempts if a.kept]
        print(f"\n  attempts (most recent):")
        for a in r.attempts[-min(5, len(r.attempts)):]:
            status = "KEPT" if a.kept else f"rev:{a.rollback_reason}"
            imp = f"{a.improvement_pct:+.1f}%" if a.improvement_pct is not None else "—"
            print(f"    [{status:>15}] {imp:>8}  {a.technique[:60]}")


def _split_kernels(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_steel",
        description="Agent Steel — agentic kernel-authoring harness for MetalBench.",
    )
    # Mode (mutually exclusive)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--kernel-name", dest="kernel_name",
        help="Optimize one or more existing kernels (comma-separated).",
    )
    mode.add_argument(
        "--new", dest="new_kernel",
        help="Author a NEW kernel from scratch (not yet implemented — placeholder).",
    )

    # Provider
    ap.add_argument(
        "--provider", default="anthropic",
        choices=list(_PROVIDER_PRESETS),
        help="LLM backend (default: anthropic). 'ollama' and 'openrouter' both "
             "use the OpenAI-compatible client with a different base_url.",
    )
    ap.add_argument("--model", default=None, help="Override the provider's default model.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM calls — use deterministic-only roofline output.")

    # Optional gputrace context (cross-check dispatched-vs-registered shape)
    ap.add_argument("--gputrace", default=None,
                    help="Path to a .gputrace bundle to enrich the diagnostic.")

    # Loop mode — full Profiler→Optimizer→Implementor→Verifier cycle
    ap.add_argument("--loop", action="store_true",
                    help="Run the full agent loop (profile → optimize → implement → verify → repeat). "
                         "Without this flag, only the Profiler runs.")
    ap.add_argument("--max-rounds", type=int, default=5,
                    help="Max loop rounds before termination (default 5).")
    ap.add_argument("--max-no-improvement", type=int, default=3,
                    help="Stop after this many consecutive rounds without a kept improvement.")

    # Parallelism
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of worker threads when --kernel-name lists multiple.")

    # Output
    ap.add_argument("--output", choices=["json", "text"], default="text")

    args = ap.parse_args(argv)

    # --new mode: placeholder for now; emit clear notice.
    if args.new_kernel:
        print(
            f"--new {args.new_kernel}: NEW-kernel authoring agent is not yet "
            "implemented. The full pipeline (Profiler→Designer→Implementor→"
            "Verifier) lands in the next milestone. Track progress in "
            "agent_steel/README.md.",
            file=sys.stderr,
        )
        return 2

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

    # Parallel — bound the pool to the smaller of (--parallel, len(kernels)).
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
