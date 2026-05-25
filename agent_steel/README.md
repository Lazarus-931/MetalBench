# Agent Steel

Closed-loop, LLM-driven harness for authoring Apple Metal compute kernels in
MetalBench. Three agents:

| Agent | LLM? | Job |
|---|---|---|
| **Profiler** | yes | `.gputrace` + bench output → chip-aware synthesizer → 2-3 paragraph diagnosis |
| **Optimizer** | yes | diagnosis + AttemptDB log + current `.metal` + MLX ref → new `.metal` to `optimizer/staging/`; accuracy-gated via `./bench` correctness; retries up to 4× on accuracy fail |
| **Verifier** | no | bench (warmup 30, iters 100), compare mean vs prior best, log ±Δ% to AttemptDB, revert on regression |

## Layout

```
agent_steel/
├── profiler/
│   ├── agent.py           # 1 LLM call → narrative
│   ├── bench_runner.py    # ./bench wrapper + cross-process GPU lock
│   ├── gputrace/          # chip-agnostic .gputrace parser
│   └── chip_metrics/
│       ├── __init__.py    # generation dispatcher
│       ├── m2.py          # active synthesizer (Xcode-CSV-shape output)
│       └── m4.py          # delegates to m2 with M4 constants
├── optimizer/
│   ├── agent.py           # LLM writes kernel + accuracy gate + retry loop
│   └── staging/           # candidate .metal files (gitignored)
├── verifier/
│   └── agent.py           # deterministic perf gate
├── loop/
│   ├── runner.py          # Profiler → Optimizer → Verifier orchestration
│   └── greedy.py          # termination strategy
├── history/
│   ├── db.py              # AttemptDB (one JSONL per kernel × chip)
│   └── models.py          # AttemptEntry schema
├── prompts/{profiler,optimizer}.md
├── providers/             # OpenAI, Anthropic, OpenAI-compat
└── __main__.py            # CLI
```

## Run

```bash
python -m agent_steel --kernel-name relu --loop --max-rounds 5
python -m agent_steel --kernel-name relu,softmax --parallel 2 --loop
python -m agent_steel --kernel-name relu --loop --provider openai --model gpt-4o
```

## AttemptDB — knowledge source

One JSONL per `(kernel, chip)` at `.agent-steel/history/<kernel>__<chip>.jsonl`.
Every entry stores the `.metal` source snapshot at that attempt, the
`gputrace_metrics` dict the Profiler saw, bench timing, the LLM's 2-3
sentence `technique` summary, and the Verifier's `kept` flag.

```python
db.read(kernel, chip)                   # full history
db.best(kernel, chip)                   # fastest kept
db.top_n_by_time(kernel, chip, n=5)     # N fastest kept
db.techniques_tried(kernel, chip)
db.failed_techniques(kernel, chip)      # (technique, rollback_reason) pairs
```

Session opens with a `technique="baseline"` entry written after the first
profile so subsequent attempts have a lineage root.

## Concurrency

| Lock | Path | Scope |
|---|---|---|
| GPU bench | `~/.agent-steel/bench.lock` | Serializes `./bench` across all agent-steel instances on the machine. |
| Session | `~/.agent-steel/locks/<kernel>__<chip>.lock` | Prevents two processes from racing on the same kernel × chip. |

## Tests

`tests/agentsteel/`:
- `test_provider.py` — LLM provider reachable
- `test_synth_artifact.py` — parser + synthesizer on a fixture `.gputrace` vs Xcode CSV
