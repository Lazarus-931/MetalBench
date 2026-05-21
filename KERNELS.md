# Kernels

Central registry of all MetalBench kernels. The website parses this file.

Format: `| name | set | metal | mlx | status |`

---

## Common Set

| name | metal | mlx | status |
|---|---|---|---|
| `sqr_mm` | [sqr_mm.metal](src/kernels/common/sqr_mm.metal) | [sqr_mm.py](mlx/kernels/common/sqr_mm.py) | ✓ |
| `rect_mm` | [rect_mm.metal](src/kernels/common/rect_mm.metal) | [rect_mm.py](mlx/kernels/common/rect_mm.py) | ✓ |
| `batch_mm` | [batch_mm.metal](src/kernels/common/batch_mm.metal) | [batch_mm.py](mlx/kernels/common/batch_mm.py) | ✓ |
| `matvec` | [matvec.metal](src/kernels/common/matvec.metal) | [matvec.py](mlx/kernels/common/matvec.py) | ✓ |
| `outer_product` | [outer_product.metal](src/kernels/common/outer_product.metal) | [outer_product.py](mlx/kernels/common/outer_product.py) | ✓ |
| `relu` | [relu.metal](src/kernels/common/relu.metal) | [relu.py](mlx/kernels/common/relu.py) | ✓ |
| `leaky_relu` | [leaky_relu.metal](src/kernels/common/leaky_relu.metal) | [leaky_relu.py](mlx/kernels/common/leaky_relu.py) | ✓ |
| `sigmoid` | [sigmoid.metal](src/kernels/common/sigmoid.metal) | [sigmoid.py](mlx/kernels/common/sigmoid.py) | ✓ |
| `swish` | [swish.metal](src/kernels/common/swish.metal) | [swish.py](mlx/kernels/common/swish.py) | ✓ |
| `gelu` | [gelu.metal](src/kernels/common/gelu.metal) | [gelu.py](mlx/kernels/common/gelu.py) | ✓ |
| `selu` | [selu.metal](src/kernels/common/selu.metal) | [selu.py](mlx/kernels/common/selu.py) | ✓ |
| `logsigmoid` | [logsigmoid.metal](src/kernels/common/logsigmoid.metal) | [logsigmoid.py](mlx/kernels/common/logsigmoid.py) | ✓ |
| `hardsigmoid` | [hardsigmoid.metal](src/kernels/common/hardsigmoid.metal) | [hardsigmoid.py](mlx/kernels/common/hardsigmoid.py) | ✓ |
| `tanh` | [tanh.metal](src/kernels/common/tanh.metal) | [tanh.py](mlx/kernels/common/tanh.py) | ✓ |
| `hardswish` | [hardswish.metal](src/kernels/common/hardswish.metal) | [hardswish.py](mlx/kernels/common/hardswish.py) | ✓ |
| `matrix_add` | [matrix_add.metal](src/kernels/common/matrix_add.metal) | [matrix_add.py](mlx/kernels/common/matrix_add.py) | ✓ |
| `matrix_scale` | [matrix_scale.metal](src/kernels/common/matrix_scale.metal) | [matrix_scale.py](mlx/kernels/common/matrix_scale.py) | ✓ |
| `dot_product` | [dot_product.metal](src/kernels/common/dot_product.metal) | [dot_product.py](mlx/kernels/common/dot_product.py) | ✓ |
| `l1_norm` | [l1_norm.metal](src/kernels/common/l1_norm.metal) | [l1_norm.py](mlx/kernels/common/l1_norm.py) | ✓ |
| `l2_norm` | [l2_norm.metal](src/kernels/common/l2_norm.metal) | [l2_norm.py](mlx/kernels/common/l2_norm.py) | ✓ |
| `layer_norm` | [layer_norm.metal](src/kernels/common/layer_norm.metal) | [layer_norm.py](mlx/kernels/common/layer_norm.py) | ✓ |
| `rms_norm` | [rms_norm.metal](src/kernels/common/rms_norm.metal) | [rms_norm.py](mlx/kernels/common/rms_norm.py) | ✓ |
| `cosine_similarity` | [cosine_similarity.metal](src/kernels/common/cosine_similarity.metal) | [cosine_similarity.py](mlx/kernels/common/cosine_similarity.py) | ✓ |
| `manhattan_similarity` | [manhattan_similarity.metal](src/kernels/common/manhattan_similarity.metal) | [manhattan_similarity.py](mlx/kernels/common/manhattan_similarity.py) | ✓ |
| `softmax` | [softmax.metal](src/kernels/common/softmax.metal) | [softmax.py](mlx/kernels/common/softmax.py) | ✓ |
| `mse_loss` | [mse_loss.metal](src/kernels/common/mse_loss.metal) | [mse_loss.py](mlx/kernels/common/mse_loss.py) | ✓ |
| `cumsum` | [cumsum.metal](src/kernels/common/cumsum.metal) | [cumsum.py](mlx/kernels/common/cumsum.py) | ✓ |
| `cumsum_reverse` | [cumsum_reverse.metal](src/kernels/common/cumsum_reverse.metal) | [cumsum_reverse.py](mlx/kernels/common/cumsum_reverse.py) | ✓ |
| `cumprod` | [cumprod.metal](src/kernels/common/cumprod.metal) | [cumprod.py](mlx/kernels/common/cumprod.py) | ✓ |
| `transpose_2d` | [transpose_2d.metal](src/kernels/common/transpose_2d.metal) | [transpose_2d.py](mlx/kernels/common/transpose_2d.py) | ✓ |
| `conv1d` | [conv1d.metal](src/kernels/common/conv1d.metal) | [conv1d.py](mlx/kernels/common/conv1d.py) | baseline |
| `conv2d` | [conv2d.metal](src/kernels/common/conv2d.metal) | [conv2d.py](mlx/kernels/common/conv2d.py) | baseline |
| `conv3d` | [conv3d.metal](src/kernels/common/conv3d.metal) | [conv3d.py](mlx/kernels/common/conv3d.py) | baseline |
| `depthwise_conv2d` | [depthwise_conv2d.metal](src/kernels/common/depthwise_conv2d.metal) | [depthwise_conv2d.py](mlx/kernels/common/depthwise_conv2d.py) | baseline |
| `conv_transpose2d` | [conv_transpose2d.metal](src/kernels/common/conv_transpose2d.metal) | [conv_transpose2d.py](mlx/kernels/common/conv_transpose2d.py) | baseline |
| `batch_norm` | [batch_norm.metal](src/kernels/common/batch_norm.metal) | [batch_norm.py](mlx/kernels/common/batch_norm.py) | ✓ |
| `exp` | [exp.metal](src/kernels/common/exp.metal) | [exp.py](mlx/kernels/common/exp.py) | stub |
| `log` | [log.metal](src/kernels/common/log.metal) | [log.py](mlx/kernels/common/log.py) | stub |
| `abs` | [abs.metal](src/kernels/common/abs.metal) | [abs.py](mlx/kernels/common/abs.py) | stub |
| `rsqrt` | [rsqrt.metal](src/kernels/common/rsqrt.metal) | [rsqrt.py](mlx/kernels/common/rsqrt.py) | stub |
| `clip` | [clip.metal](src/kernels/common/clip.metal) | [clip.py](mlx/kernels/common/clip.py) | stub |
| `elu` | [elu.metal](src/kernels/common/elu.metal) | [elu.py](mlx/kernels/common/elu.py) | ✓ |
| `mish` | [mish.metal](src/kernels/common/mish.metal) | [mish.py](mlx/kernels/common/mish.py) | ✓ |
| `softplus` | [softplus.metal](src/kernels/common/softplus.metal) | [softplus.py](mlx/kernels/common/softplus.py) | ✓ |
| `prelu` | [prelu.metal](src/kernels/common/prelu.metal) | [prelu.py](mlx/kernels/common/prelu.py) | ✓ |
| `argmax` | [argmax.metal](src/kernels/common/argmax.metal) | [argmax.py](mlx/kernels/common/argmax.py) | ✓ |
| `variance` | [variance.metal](src/kernels/common/variance.metal) | [variance.py](mlx/kernels/common/variance.py) | ✓ |

## Standard Set

### Done (5)

| name | metal | mlx | status |
|---|---|---|---|
| `add_norm` | [add_norm.metal](src/kernels/standard/add_norm.metal) | [add_norm.py](mlx/kernels/standard/add_norm.py) | ✓ |
| `silu_linear` | [silu_linear.metal](src/kernels/standard/silu_linear.metal) | [silu_linear.py](mlx/kernels/standard/silu_linear.py) | ✓ |
| `gelu_linear` | [gelu_linear.metal](src/kernels/standard/gelu_linear.metal) | [gelu_linear.py](mlx/kernels/standard/gelu_linear.py) | debug |
| `rms_norm_linear` | [rms_norm_linear.metal](src/kernels/standard/rms_norm_linear.metal) | [rms_norm_linear.py](mlx/kernels/standard/rms_norm_linear.py) | debug |
| `scaled_dot_product` | [scaled_dot_product.metal](src/kernels/standard/scaled_dot_product.metal) | [scaled_dot_product.py](mlx/kernels/standard/scaled_dot_product.py) | WIP |
| `rope_embedding` | [rope_embedding.metal](src/kernels/standard/rope_embedding.metal) | [rope_embedding.py](mlx/kernels/standard/rope_embedding.py) | ✓ |
| `swiglu` | [swiglu.metal](src/kernels/standard/swiglu.metal) | [swiglu.py](mlx/kernels/standard/swiglu.py) | ✓ |
| `softmax_attention` | [softmax_attention.metal](src/kernels/standard/softmax_attention.metal) | [softmax_attention.py](mlx/kernels/standard/softmax_attention.py) | ✓ |
| `residual_add` | [residual_add.metal](src/kernels/standard/residual_add.metal) | [residual_add.py](mlx/kernels/standard/residual_add.py) | ✓ |
| `dropout` | [dropout.metal](src/kernels/standard/dropout.metal) | [dropout.py](mlx/kernels/standard/dropout.py) | ✓ |
| `instance_norm` | [instance_norm.metal](src/kernels/standard/instance_norm.metal) | [instance_norm.py](mlx/kernels/standard/instance_norm.py) | ✓ |
| `group_norm` | [group_norm.metal](src/kernels/standard/group_norm.metal) | [group_norm.py](mlx/kernels/standard/group_norm.py) | ✓ |
| `cross_entropy_loss` | [cross_entropy_loss.metal](src/kernels/standard/cross_entropy_loss.metal) | [cross_entropy_loss.py](mlx/kernels/standard/cross_entropy_loss.py) | ✓ |
| `log_softmax` | [log_softmax.metal](src/kernels/standard/log_softmax.metal) | [log_softmax.py](mlx/kernels/standard/log_softmax.py) | ✓ |
| `masked_softmax` | [masked_softmax.metal](src/kernels/standard/masked_softmax.metal) | [masked_softmax.py](mlx/kernels/standard/masked_softmax.py) | ✓ |
| `bias_add` | [bias_add.metal](src/kernels/standard/bias_add.metal) | [bias_add.py](mlx/kernels/standard/bias_add.py) | ✓ |
| `fused_add_rms_norm` | [fused_add_rms_norm.metal](src/kernels/standard/fused_add_rms_norm.metal) | [fused_add_rms_norm.py](mlx/kernels/standard/fused_add_rms_norm.py) | ✓ |
| `linear_bias` | [linear_bias.metal](src/kernels/standard/linear_bias.metal) | [linear_bias.py](mlx/kernels/standard/linear_bias.py) | ✓ |
| `bias_gelu` | [bias_gelu.metal](src/kernels/standard/bias_gelu.metal) | [bias_gelu.py](mlx/kernels/standard/bias_gelu.py) | ✓ |
| `fused_qkv_projection` | [fused_qkv_projection.metal](src/kernels/standard/fused_qkv_projection.metal) | [fused_qkv_projection.py](mlx/kernels/standard/fused_qkv_projection.py) | ✓ |
| `attention_scores` | [attention_scores.metal](src/kernels/standard/attention_scores.metal) | [attention_scores.py](mlx/kernels/standard/attention_scores.py) | ✓ |
| `nll_loss` | [nll_loss.metal](src/kernels/standard/nll_loss.metal) | [nll_loss.py](mlx/kernels/standard/nll_loss.py) | ✓ |
| `log_softmax_cross_entropy` | [log_softmax_cross_entropy.metal](src/kernels/standard/log_softmax_cross_entropy.metal) | [log_softmax_cross_entropy.py](mlx/kernels/standard/log_softmax_cross_entropy.py) | ✓ |

### Planned (10)

| # | name | op | shape | models |
|---|---|---|---|---|
| 1 | `rope_embedding` | rotary position embedding: x * cos + rotate(x) * sin | (128, 64) | LLaMA, Mistral, Gemma, Qwen |
| 2 | `swiglu` | SiLU-gated linear unit: silu(x @ W_gate) * (x @ W_up) | (1024, 1024) | LLaMA, Mistral, Gemma |
| 4 | `softmax_attention` | Q @ K^T / √d → softmax → @ V | (128, 128) | all transformers |
| 5 | `residual_add` | y = x + residual (with optional scale α) | (1024, 1024) | every residual network |
| 6 | `dropout` | training-mode dropout: mask * x / (1-p) | (1024, 1024) | training only |
| 8 | `instance_norm` | μ,σ per sample per channel | (128, 256, 64, 64) | style transfer, GANs |
| 9 | `group_norm` | μ,σ per group of channels | (128, 256, 64, 64) | detection, segmentation |
| 10 | `cross_entropy_loss` | -Σ y_true * log(softmax(y_pred)) | (1024, 1024) | classification training |

### Full Set (future)

| name | op |
|---|---|
| `transformer_block` | full transformer block (attn + FFN + norms) |
| `llama_attention` | grouped query attention + RoPE + KV cache |
| `conv_block` | conv2d + batchnorm + relu fused |
| `mbconv` | MobileNet inverted residual block |
