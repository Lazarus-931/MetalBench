# Kernels

Concrete benchmark list. See [README](README.md) for the Common / Standard / Full split.

Every entry maps 1:1 to two files:

| file | role |
|---|---|
| `mlx/kernels/<set>/<name>.py` | MLX baseline (Model class only) |
| `src/kernels/<set>/<name>.metal` | Metal kernel |

Metadata lives in `<set>/registry.py`. Run any kernel with `./bench <name>`.

---

## Common Set — 30+ kernels

### Matmul ops
| name | op |
|---|---|
| `sqr_mm` | `A @ B` (square N×N) |
| `rect_mm` | `A @ B` (M×K @ K×N) |
| `batch_mm` | `A_b @ B_b` (batched) |
| `matvec` | `A @ x` |
| `outer_product` | `x y^T` |

### Element-wise activations
| name | op |
|---|---|
| `relu` | `max(x, 0)` |
| `leaky_relu` | `max(x, 0) + slope * min(x, 0)` |
| `sigmoid` | `1 / (1 + exp(-x))` |
| `swish` | `x * sigmoid(x)` |
| `gelu` | GELU activation |
| `selu` | SELU activation |
| `logsigmoid` | `log(sigmoid(x))` |
| `hardsigmoid` | `clamp(x/6 + 0.5, 0, 1)` |
| `tanh` | `tanh(x)` |
| `hardswish` | `x * clamp(x+3, 0, 6) / 6` |

### Element-wise arithmetic
| name | op |
|---|---|
| `matrix_add` | `A + B` |
| `matrix_scale` | `alpha * A` |

### Reductions
| name | op |
|---|---|
| `dot_product` | `x^T y` |
| `l1_norm` | `sum(abs(x))` along last dim |
| `l2_norm` | `sqrt(sum(x^2))` along last dim |
| `trace` | `sum(A_ii)` |
| `mse_loss` | `mean((pred - target)^2)` |
| `softmax` | `exp(x) / sum(exp(x))` per row |
| `cosine_similarity` | `x·y / (|x|·|y|)` per row pair |
| `manhattan_similarity` | `sum(|x - y|)` per row pair |

### Normalization
| name | op |
|---|---|
| `layer_norm` | `(x - mean) / sqrt(var + eps)` |
| `rms_norm` | `x * rsqrt(mean(x^2) + eps)` |

### Scans
| name | op |
|---|---|
| `cumsum` | cumulative sum along last dim |
| `cumsum_reverse` | reverse cumulative sum |
| `cumprod` | cumulative product along last dim |

### Misc
| name | op |
|---|---|
| `transpose_2d` | `A^T` |
| `argmax` (planned) | index of max per row |

### Convolutions (Metal kernels WIP)
| name | op |
|---|---|
| `conv1d` | 1D convolution |
| `conv2d` | 2D convolution |
| `conv3d` | 3D convolution |
| `depthwise_conv2d` | depthwise 2D convolution |
| `conv_transpose2d` | transposed 2D convolution |

---

## Standard Set — fused kernels (5, expanding)

| name | op | status |
|---|---|---|
| `add_norm` | `layer_norm(x + residual)` | ✓ |
| `silu_linear` | `silu(x @ W)` | ✓ |
| `gelu_linear` | `gelu(x @ W)` | debugging |
| `rms_norm_linear` | `rms_norm(x) @ W` | debugging |
| `scaled_dot_product` | `softmax(Q@K^T / sqrt(d)) @ V` | WIP |

---

## Full Set

Coming — multi-op kernels at architecture scale (full attention blocks, MLP blocks, conv layers).
