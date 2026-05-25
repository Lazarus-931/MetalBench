Chip: <!-- m2 / m4 / etc. — which M-chip generation you measured this on. Required for `./verify` to compare on the same hardware. -->

SOL: <!-- e.g. `38% of M2's peak compute` or `82% of M2's peak BW` — whichever the kernel's bottleneck is. Pull from `chip_aware_metrics.sol_compute_pct` / `sol_bw_pct` in agent-steel's output, or compute as achieved_GFLOPS / peak_TFLOPS (for compute-bound) or achieved_GBps / peak_GBps (for memory-bound). -->

## What

<!-- Which kernel(s) did you change? What did you change? -->

## Results

<!-- Paste the output of ./bench <name> showing correctness and speedup -->

```
```

## Checklist

- [ ] `./bench <name>` returns `correct=true`
- [ ] Only `.metal` file(s) (and possibly `mlx/kernels/.../registry.py` for dispatch shape changes) changed
- [ ] Tested on my local machine
- [ ] If this is a chip-specific optimization, the change is in `metal/kernels/<set>/<kernel>/<chip>.metal` (not shared `default.metal` or flat). See CONTRIBUTING.md.

> `best_times.md` and `LINK.md` are **auto-generated** from `session.json` — do not hand-edit them in your PR.
