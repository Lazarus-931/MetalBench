# Optimize a single kernel — agent prompt template

Variables (caller fills in with simple `{{name}}` substitution):
- `{{kernel_name}}` — e.g. `softmax_attention`
- `{{set}}` — `common` / `standard` / `full`
- `{{chip_target}}` — `m2` / `m4` / etc.
- `{{baseline_speedup}}` — current `./bench` speedup
- `{{bench_cmd}}` — `./bench` or `./bench_remote` or `/tmp/MetalBench-<host>/bench_remote`
- `{{working_dir}}` — `/Users/alazarmanakelew/MetalBench` or a per-mini path

---

You are optimizing ONE kernel — `{{kernel_name}}` — for Apple {{chip_target}}.

Working directory: `{{working_dir}}`. Bench via `{{bench_cmd}} {{kernel_name}}`.

Current speedup: `{{baseline_speedup}}` vs MLX.

## Workflow
1. Read `mlx/kernels/{{set}}/{{kernel_name}}.py` for the spec — DO NOT edit.
2. Read `mlx/kernels/{{set}}/registry.py` for input bindings / shapes / scalars.
3. Read the current `metal/kernels/{{set}}/{{kernel_name}}.metal` (or `<name>/<chip>.metal` if it's a directory variant) — understand why it's slow.
4. Bench baseline 3× via `{{bench_cmd}} {{kernel_name}}`; take median. That's your "before" number.
5. Edit the .metal file. Bench 3×. If median improves AND `correctness : ✓ correct`, keep. Otherwise revert.
6. Budget 2–3 attempts. If no improvement after that, leave it and move on.

## Hard rules
- Use `{{bench_cmd}}` exclusively.
- Never edit registry.py, harness, host C++, Makefile, or .py baselines.
- Buffer bindings MUST match what registry says.
- Require `correctness : ✓ correct` on every kept change. Revert if broken.
- Median of 3 runs only. No single-run claims.

## What the roofline line in bench output tells you
- `memory-bound` → try float4 grid-stride, larger grid, fewer device reads
- `compute-bound` → simdgroup_matrix MMA tiles, inner-loop unroll, FMA fusion
- `latency-dominated` → kernel-launch overhead is the floor; structural changes only

## Report format
```
KERNEL              BEFORE   AFTER    STATUS
{{kernel_name}}     X.XXx    Y.YYx    improved (one-line design note)
```
