# Kernels

Central registry of all MetalBench kernels. The website parses this file.

Each row links to the Metal kernel and the MLX reference. When a kernel
directory contains chip variants (e.g. `m4.metal`, `m4_m5.metal`), the
link points at the directory — the dashboard resolves the right variant
per chip at click-time. See [LINK.md](results/) for the best-known
variant per kernel per chip.

---

## Common Set (75)

| name | metal | mlx |
|---|---|---|
| `sqr_mm` | [metal](src/kernels/common/sqr_mm.metal) | [mlx](mlx/kernels/common/sqr_mm.py) |
| `rect_mm` | [metal](src/kernels/common/rect_mm/) | [mlx](mlx/kernels/common/rect_mm.py) |
| `batch_mm` | [metal](src/kernels/common/batch_mm/) | [mlx](mlx/kernels/common/batch_mm.py) |
| `matvec` | [metal](src/kernels/common/matvec.metal) | [mlx](mlx/kernels/common/matvec.py) |
| `outer_product` | [metal](src/kernels/common/outer_product.metal) | [mlx](mlx/kernels/common/outer_product.py) |
| `relu` | [metal](src/kernels/common/relu.metal) | [mlx](mlx/kernels/common/relu.py) |
| `leaky_relu` | [metal](src/kernels/common/leaky_relu.metal) | [mlx](mlx/kernels/common/leaky_relu.py) |
| `prelu` | [metal](src/kernels/common/prelu.metal) | [mlx](mlx/kernels/common/prelu.py) |
| `sigmoid` | [metal](src/kernels/common/sigmoid.metal) | [mlx](mlx/kernels/common/sigmoid.py) |
| `hardsigmoid` | [metal](src/kernels/common/hardsigmoid.metal) | [mlx](mlx/kernels/common/hardsigmoid.py) |
| `logsigmoid` | [metal](src/kernels/common/logsigmoid.metal) | [mlx](mlx/kernels/common/logsigmoid.py) |
| `tanh` | [metal](src/kernels/common/tanh.metal) | [mlx](mlx/kernels/common/tanh.py) |
| `hardtanh` | [metal](src/kernels/common/hardtanh.metal) | [mlx](mlx/kernels/common/hardtanh.py) |
| `swish` | [metal](src/kernels/common/swish.metal) | [mlx](mlx/kernels/common/swish.py) |
| `hardswish` | [metal](src/kernels/common/hardswish.metal) | [mlx](mlx/kernels/common/hardswish.py) |
| `gelu` | [metal](src/kernels/common/gelu.metal) | [mlx](mlx/kernels/common/gelu.py) |
| `mingpt_new_gelu` | [metal](src/kernels/common/mingpt_new_gelu.metal) | [mlx](mlx/kernels/common/mingpt_new_gelu.py) |
| `selu` | [metal](src/kernels/common/selu.metal) | [mlx](mlx/kernels/common/selu.py) |
| `elu` | [metal](src/kernels/common/elu.metal) | [mlx](mlx/kernels/common/elu.py) |
| `mish` | [metal](src/kernels/common/mish.metal) | [mlx](mlx/kernels/common/mish.py) |
| `softplus` | [metal](src/kernels/common/softplus.metal) | [mlx](mlx/kernels/common/softplus.py) |
| `softsign` | [metal](src/kernels/common/softsign.metal) | [mlx](mlx/kernels/common/softsign.py) |
| `softmax` | [metal](src/kernels/common/softmax.metal) | [mlx](mlx/kernels/common/softmax.py) |
| `abs` | [metal](src/kernels/common/abs.metal) | [mlx](mlx/kernels/common/abs.py) |
| `exp` | [metal](src/kernels/common/exp.metal) | [mlx](mlx/kernels/common/exp.py) |
| `log` | [metal](src/kernels/common/log.metal) | [mlx](mlx/kernels/common/log.py) |
| `rsqrt` | [metal](src/kernels/common/rsqrt.metal) | [mlx](mlx/kernels/common/rsqrt.py) |
| `clip` | [metal](src/kernels/common/clip.metal) | [mlx](mlx/kernels/common/clip.py) |
| `where` | [metal](src/kernels/common/where.metal) | [mlx](mlx/kernels/common/where.py) |
| `embedding` | [metal](src/kernels/common/embedding.metal) | [mlx](mlx/kernels/common/embedding.py) |
| `matrix_add` | [metal](src/kernels/common/matrix_add.metal) | [mlx](mlx/kernels/common/matrix_add.py) |
| `matrix_scale` | [metal](src/kernels/common/matrix_scale.metal) | [mlx](mlx/kernels/common/matrix_scale.py) |
| `dot_product` | [metal](src/kernels/common/dot_product.metal) | [mlx](mlx/kernels/common/dot_product.py) |
| `cosine_similarity` | [metal](src/kernels/common/cosine_similarity.metal) | [mlx](mlx/kernels/common/cosine_similarity.py) |
| `manhattan_similarity` | [metal](src/kernels/common/manhattan_similarity.metal) | [mlx](mlx/kernels/common/manhattan_similarity.py) |
| `transpose_2d` | [metal](src/kernels/common/transpose_2d.metal) | [mlx](mlx/kernels/common/transpose_2d.py) |
| `l1_norm` | [metal](src/kernels/common/l1_norm.metal) | [mlx](mlx/kernels/common/l1_norm.py) |
| `l2_norm` | [metal](src/kernels/common/l2_norm.metal) | [mlx](mlx/kernels/common/l2_norm.py) |
| `frobenius_norm` | [metal](src/kernels/common/frobenius_norm.metal) | [mlx](mlx/kernels/common/frobenius_norm.py) |
| `layer_norm` | [metal](src/kernels/common/layer_norm/) | [mlx](mlx/kernels/common/layer_norm.py) |
| `rms_norm` | [metal](src/kernels/common/rms_norm.metal) | [mlx](mlx/kernels/common/rms_norm.py) |
| `batch_norm` | [metal](src/kernels/common/batch_norm.metal) | [mlx](mlx/kernels/common/batch_norm.py) |
| `variance` | [metal](src/kernels/common/variance.metal) | [mlx](mlx/kernels/common/variance.py) |
| `argmax` | [metal](src/kernels/common/argmax.metal) | [mlx](mlx/kernels/common/argmax.py) |
| `top_k` | [metal](src/kernels/common/top_k/) | [mlx](mlx/kernels/common/top_k.py) |
| `logsumexp` | [metal](src/kernels/common/logsumexp/) | [mlx](mlx/kernels/common/logsumexp.py) |
| `cumsum` | [metal](src/kernels/common/cumsum.metal) | [mlx](mlx/kernels/common/cumsum.py) |
| `cumsum_reverse` | [metal](src/kernels/common/cumsum_reverse.metal) | [mlx](mlx/kernels/common/cumsum_reverse.py) |
| `cumsum_exclusive` | [metal](src/kernels/common/cumsum_exclusive.metal) | [mlx](mlx/kernels/common/cumsum_exclusive.py) |
| `cumprod` | [metal](src/kernels/common/cumprod.metal) | [mlx](mlx/kernels/common/cumprod.py) |
| `masked_cumsum` | [metal](src/kernels/common/masked_cumsum.metal) | [mlx](mlx/kernels/common/masked_cumsum.py) |
| `mse_loss` | [metal](src/kernels/common/mse_loss.metal) | [mlx](mlx/kernels/common/mse_loss.py) |
| `hinge_loss` | [metal](src/kernels/common/hinge_loss.metal) | [mlx](mlx/kernels/common/hinge_loss.py) |
| `huber_loss` | [metal](src/kernels/common/huber_loss/) | [mlx](mlx/kernels/common/huber_loss.py) |
| `kl_div_loss` | [metal](src/kernels/common/kl_div_loss/) | [mlx](mlx/kernels/common/kl_div_loss.py) |
| `triplet_margin_loss` | [metal](src/kernels/common/triplet_margin_loss.metal) | [mlx](mlx/kernels/common/triplet_margin_loss.py) |
| `conv1d` | [metal](src/kernels/common/conv1d/) | [mlx](mlx/kernels/common/conv1d.py) |
| `conv2d` | [metal](src/kernels/common/conv2d/) | [mlx](mlx/kernels/common/conv2d.py) |
| `conv3d` | [metal](src/kernels/common/conv3d/) | [mlx](mlx/kernels/common/conv3d.py) |
| `depthwise_conv2d` | [metal](src/kernels/common/depthwise_conv2d.metal) | [mlx](mlx/kernels/common/depthwise_conv2d.py) |
| `conv_transpose2d` | [metal](src/kernels/common/conv_transpose2d/) | [mlx](mlx/kernels/common/conv_transpose2d.py) |
| `conv2d_relu_bias` | [metal](src/kernels/common/conv2d_relu_bias.metal) | [mlx](mlx/kernels/common/conv2d_relu_bias.py) |
| `conv2d_mish_mish` | [metal](src/kernels/common/conv2d_mish_mish.metal) | [mlx](mlx/kernels/common/conv2d_mish_mish.py) |
| `conv3d_div_pool_sum` | [metal](src/kernels/common/conv3d_div_pool_sum.metal) | [mlx](mlx/kernels/common/conv3d_div_pool_sum.py) |
| `conv3d_softmax_pool` | [metal](src/kernels/common/conv3d_softmax_pool.metal) | [mlx](mlx/kernels/common/conv3d_softmax_pool.py) |
| `conv3d_multi_act_bias` | [metal](src/kernels/common/conv3d_multi_act_bias.metal) | [mlx](mlx/kernels/common/conv3d_multi_act_bias.py) |
| `conv_transpose2d_clamp_scale_div` | [metal](src/kernels/common/conv_transpose2d_clamp_scale_div.metal) | [mlx](mlx/kernels/common/conv_transpose2d_clamp_scale_div.py) |
| `conv_transpose2d_sub_tanh` | [metal](src/kernels/common/conv_transpose2d_sub_tanh.metal) | [mlx](mlx/kernels/common/conv_transpose2d_sub_tanh.py) |
| `conv_transpose3d_norm_pool_gelu` | [metal](src/kernels/common/conv_transpose3d_norm_pool_gelu.metal) | [mlx](mlx/kernels/common/conv_transpose3d_norm_pool_gelu.py) |
| `matmul_sub_mul_relu` | [metal](src/kernels/common/matmul_sub_mul_relu.metal) | [mlx](mlx/kernels/common/matmul_sub_mul_relu.py) |
| `avg_pool1d` | [metal](src/kernels/common/avg_pool1d.metal) | [mlx](mlx/kernels/common/avg_pool1d.py) |
| `avg_pool2d` | [metal](src/kernels/common/avg_pool2d.metal) | [mlx](mlx/kernels/common/avg_pool2d.py) |
| `avg_pool3d` | [metal](src/kernels/common/avg_pool3d.metal) | [mlx](mlx/kernels/common/avg_pool3d.py) |
| `max_pool1d` | [metal](src/kernels/common/max_pool1d.metal) | [mlx](mlx/kernels/common/max_pool1d.py) |
| `max_pool2d` | [metal](src/kernels/common/max_pool2d.metal) | [mlx](mlx/kernels/common/max_pool2d.py) |
| `max_pool3d` | [metal](src/kernels/common/max_pool3d/) | [mlx](mlx/kernels/common/max_pool3d.py) |

## Standard Set (25)

| name | metal | mlx |
|---|---|---|
| `add_norm` | [metal](src/kernels/standard/add_norm.metal) | [mlx](mlx/kernels/standard/add_norm.py) |
| `silu_linear` | [metal](src/kernels/standard/silu_linear.metal) | [mlx](mlx/kernels/standard/silu_linear.py) |
| `gelu_linear` | [metal](src/kernels/standard/gelu_linear.metal) | [mlx](mlx/kernels/standard/gelu_linear.py) |
| `rms_norm_linear` | [metal](src/kernels/standard/rms_norm_linear.metal) | [mlx](mlx/kernels/standard/rms_norm_linear.py) |
| `scaled_dot_product` | [metal](src/kernels/standard/scaled_dot_product.metal) | [mlx](mlx/kernels/standard/scaled_dot_product.py) |
| `rope_embedding` | [metal](src/kernels/standard/rope_embedding.metal) | [mlx](mlx/kernels/standard/rope_embedding.py) |
| `swiglu` | [metal](src/kernels/standard/swiglu.metal) | [mlx](mlx/kernels/standard/swiglu.py) |
| `softmax_attention` | [metal](src/kernels/standard/softmax_attention.metal) | [mlx](mlx/kernels/standard/softmax_attention.py) |
| `residual_add` | [metal](src/kernels/standard/residual_add.metal) | [mlx](mlx/kernels/standard/residual_add.py) |
| `dropout` | [metal](src/kernels/standard/dropout.metal) | [mlx](mlx/kernels/standard/dropout.py) |
| `instance_norm` | [metal](src/kernels/standard/instance_norm.metal) | [mlx](mlx/kernels/standard/instance_norm.py) |
| `group_norm` | [metal](src/kernels/standard/group_norm.metal) | [mlx](mlx/kernels/standard/group_norm.py) |
| `cross_entropy_loss` | [metal](src/kernels/standard/cross_entropy_loss.metal) | [mlx](mlx/kernels/standard/cross_entropy_loss.py) |
| `log_softmax` | [metal](src/kernels/standard/log_softmax.metal) | [mlx](mlx/kernels/standard/log_softmax.py) |
| `masked_softmax` | [metal](src/kernels/standard/masked_softmax.metal) | [mlx](mlx/kernels/standard/masked_softmax.py) |
| `bias_add` | [metal](src/kernels/standard/bias_add.metal) | [mlx](mlx/kernels/standard/bias_add.py) |
| `bias_gelu` | [metal](src/kernels/standard/bias_gelu.metal) | [mlx](mlx/kernels/standard/bias_gelu.py) |
| `fused_add_rms_norm` | [metal](src/kernels/standard/fused_add_rms_norm.metal) | [mlx](mlx/kernels/standard/fused_add_rms_norm.py) |
| `linear_bias` | [metal](src/kernels/standard/linear_bias.metal) | [mlx](mlx/kernels/standard/linear_bias.py) |
| `fused_qkv_projection` | [metal](src/kernels/standard/fused_qkv_projection.metal) | [mlx](mlx/kernels/standard/fused_qkv_projection.py) |
| `attention_scores` | [metal](src/kernels/standard/attention_scores.metal) | [mlx](mlx/kernels/standard/attention_scores.py) |
| `nll_loss` | [metal](src/kernels/standard/nll_loss.metal) | [mlx](mlx/kernels/standard/nll_loss.py) |
| `log_softmax_cross_entropy` | [metal](src/kernels/standard/log_softmax_cross_entropy.metal) | [mlx](mlx/kernels/standard/log_softmax_cross_entropy.py) |
| `llama_attention` | [metal](src/kernels/standard/llama_attention.metal) | [mlx](mlx/kernels/standard/llama_attention.py) |
| `matmul_gelu_softmax` | [metal](src/kernels/standard/matmul_gelu_softmax.metal) | [mlx](mlx/kernels/standard/matmul_gelu_softmax.py) |

## Full Set (6)

| name | metal | mlx |
|---|---|---|
| `alexnet` | [metal](src/kernels/full/alexnet.metal) | [mlx](mlx/kernels/full/alexnet.py) |
| `mlp` | [metal](src/kernels/full/mlp.metal) | [mlx](mlx/kernels/full/mlp.py) |
| `resnet` | [metal](src/kernels/full/resnet.metal) | [mlx](mlx/kernels/full/resnet.py) |
| `llama_decoder_layer` | [metal](src/kernels/full/llama_decoder_layer.metal) | [mlx](mlx/kernels/full/llama_decoder_layer.py) |
| `transformer_block` | [metal](src/kernels/full/transformer_block.metal) | [mlx](mlx/kernels/full/transformer_block.py) |
| `densenet` | _scaffold pending_ | _scaffold pending_ |
