# Adding a new Apple M-series chip generation

MetalBench keeps **all** chip metadata in `chips.json` at the repository root.
Adding a new generation (M6, M7, ...) is a one-file edit. The Python registry
(`agent_steel/chips.py`) reads `chips.json` at import time; the C++ host reads
a header (`metal/scripts/chip_table.h`) that the `Makefile` regenerates from
`chips.json` automatically.

## The literal diff to add M6

```diff
--- a/chips.json
+++ b/chips.json
@@
   "chips": [
+    {
+      "gen": "m6",
+      "brand": "M6",
+      "variants": ["Ultra", "Max", "Pro", ""],
+      "peak_compute_TFLOPS": 6.5,
+      "peak_bandwidth_GBps": 180.0,
+      "tg_memory_max_bytes": 32768
+    },
     {
       "gen": "m5",
       "brand": "M5",
```

That's it. Run `make kernels host` and every consumer below picks up M6:

- C++ `MChipType::M6 / M6_PRO / M6_MAX / M6_ULTRA` — generated into the enum
  via the X-macro in `chip_table.h`.
- C++ `type_name()` switch — same X-macro.
- C++ `parse_type()` ladder — same X-macro (correct most-specific-first order).
- Python `mlx_helpers._CHIP_TYPES` — derived via `chips.list_chip_types()`.
- Python `roofline.CHIP_PEAKS` and `_generation()` — derived via `chips.CHIPS`.
- Python `gputrace_check._CHIP_CEILINGS` — derived via `chips.ceiling()`.
- `agent_steel.profiler.agent`, `implementor.agent`, `loop.runner`,
  `gputrace_check._chip_id` — all call `chips.detect_generation()`.
- `agent_steel/profiler/chip_metrics/m6.py` — only needed if you want
  per-variant core / TFLOPS / bandwidth specs beyond the base-chip numbers;
  copy `m4.py` as a template. Auto-discovered by `pkgutil.iter_modules()`.

## Variant ordering

Entries within `variants` must be most-specific-first (e.g. `["Ultra", "Max",
"Pro", ""]`), and chips within `chips.json` must be newest-first, because
`parse_type()` returns the first match. The C++ header generator preserves
the JSON order verbatim.

## Verifying

```bash
python3 tests/test_chip_registry.py    # registry smoke tests
make kernels host                       # rebuilds chip_table.h + host binary
./build/metalbench_host --list-chip     # prints detected chip JSON
```
