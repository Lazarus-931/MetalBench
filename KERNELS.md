# Kernels

Concrete benchmark list. See [README](README.md) for the Common / Standard / Full split and overall philosophy.

Every entry below maps 1:1 to two files:

| file | role |
|---|---|
| `python/metalbench/<name>.py` | MLX baseline + correctness reference |
| `src/kernels/<name>/kernel.metal` | candidate Metal kernel |

Run any one with `./bench <name>`. Results land in `results/<chip-bucket>/<name>.json` so different M-chips don't get mashed into the same numbers.

---

## Common Set — first 50

The Common Set targets 100 simple operations + the building blocks of larger kernels. These first 50 are the foundation — get every one of these green before reaching for fused/full kernels.

### Matrix ops (start here — what the README leads with)
| # | name | op |
|---|---|---|
| 1 | `sqr_matmul`     | `A @ B` (square N×N) |
| 2 | `rect_matmul`    | `A @ B` (M×K @ K×N) |
| 3 | `batched_matmul` | `A_b @ B_b` |
| 4 | `matvec`         | `A @ x` |
| 5 | `transpose_2d`   | `Aᵀ` |
| 6 | `dot_product`    | `xᵀ y` |
| 7 | `outer_product`  | `x yᵀ` |
| 8 | `matrix_add`     | `A + B` |
| 9 | `matrix_scale`   | `α · A` |
| 10 | `trace`          | `Σ A_ii` |






---

## Coming next

- **Common Set 51-100** — the remaining 50 unary/binary/reduction/matrix variants (different dtypes, axes, broadcasting).
- **Standard Set (50)** — fused kernels: `gelu_linear`, `softmax_attention`, `rms_norm_linear`, etc.
- **Full Set (25)** — multi-op kernels at architecture scale: full attention block, MLP block, conv layers.
