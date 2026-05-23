# Verify a PR — reviewer prompt template

Used by `./verify <pr>` (or a reviewer LLM looking at PR diff + bench output) to
turn raw measurements into a PASS/FAIL decision with reasoning.

Variables:
- `{{pr_id}}`
- `{{chip}}` — reviewer's chip
- `{{kernel_name}}`
- `{{claimed_ms}}`, `{{claimed_speedup}}`
- `{{measured_ms}}`, `{{measured_speedup}}`
- `{{correct}}` — `✓` or `✗`
- `{{tolerance_pct}}` — e.g. `15`
- `{{metal_diff}}` — `git diff` of the .metal file

---

Verify PR #{{pr_id}} claim for `{{kernel_name}}` on chip `{{chip}}`.

## Claim
- {{claimed_ms}} ms · {{claimed_speedup}}× vs MLX

## My measurement (median of 3 runs)
- {{measured_ms}} ms · {{measured_speedup}}× vs MLX · correctness {{correct}}

## Tolerance
±{{tolerance_pct}}% on time.

## Diff
```diff
{{metal_diff}}
```

## Required output
Three short lines:
1. **VERDICT:** `VERIFIED` / `REJECTED` / `INCONCLUSIVE`.
2. Reason (one sentence — e.g. "measured 0.092ms vs claimed 0.087ms, within tolerance" or "kernel imports MLX, cheats the correctness check").
3. Reviewer-facing copy/paste line for the PR comment, e.g.:
   `{{kernel_name}}: claimed {{claimed_speedup}}×, measured {{measured_speedup}}× on {{chip}} — VERIFIED ✓`
