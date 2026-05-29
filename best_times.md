# MetalBench Best Times

Best kernel time + speedup vs MLX per chip. `—` = not yet benchmarked on that chip.
Auto-generated from `session.json` by `scripts/render_best_times.py`. Do not hand-edit.

| kernel | set | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| `abs` | common | — | 0.018 (454.63×) | — | 0.018 (380.45×) | — |
| `argmax` | common | — | 0.072 (5.10×) | — | 4.025 (1.99×) | — |
| `avg_pool1d` | common | — | 0.013 (46.84×) | — | 2.999 (2.33×) | — |
| `avg_pool2d` | common | — | 0.124 (4.02×) | — | 4.007 (2.00×) | — |
| `avg_pool3d` | common | — | 0.223 (5.26×) | — | 7.007 (1.14×) | — |
| `batch_mm` | common | — | 2.693 (0.91×) | — | 128.046 (0.08×) | — |
| `batch_norm` | common | — | 0.068 (7.00×) | — | 5.011 (2.20×) | — |
| `clip` | common | — | 0.010 (20.03×) | — | 3.005 (2.33×) | — |
| `conv1d` | common | — | 0.137 (2.40×) | — | 5.011 (1.39×) | — |
| `conv2d` | common | — | 3.517 (0.82×) | — | 8.748 (1.24×) | — |
| `conv2d_mish_mish` | common | — | 5.833 (1.08×) | — | 11.556 (33.23×) | — |
| `conv2d_relu_bias` | common | — | 518.741 (0.02×) | — | 11.406 (7.45×) | — |
| `conv3d` | common | — | 15.365 (0.46×) | — | 118.961 (0.20×) | — |
| `conv3d_div_pool_sum` | common | — | 46.736 (0.17×) | — | 189.517 (0.55×) | — |
| `conv3d_multi_act_bias` | common | — | 19.095 (0.63×) | — | 31.664 (16.93×) | — |
| `conv3d_softmax_pool` | common | — | 6.640 (1.30×) | — | 21.227 (6.17×) | — |
| `conv_transpose2d` | common | — | 8.878 (0.16×) | — | 13.143 (0.61×) | — |
| `conv_transpose2d_clamp_scale_div` | common | — | 160.275 (0.04×) | — | 11.533 (25.92×) | — |
| `conv_transpose2d_sub_tanh` | common | — | 9.610 (0.23×) | — | 11.537 (9.01×) | — |
| `conv_transpose3d_norm_pool_gelu` | common | — | 0.674 (0.87×) | — | 3.186 (4.08×) | — |
| `cosine_similarity` | common | — | 0.069 (11.29×) | — | 8.038 (4.98×) | — |
| `cumprod` | common | — | 0.078 (5.16×) | — | 4.019 (1.99×) | — |
| `cumsum` | common | — | 0.097 (3.07×) | — | 10.899 (0.73×) | — |
| `cumsum_exclusive` | common | — | 0.059 (7.32×) | — | 11.014 (1.82×) | — |
| `cumsum_reverse` | common | — | 0.102 (3.91×) | — | 8.021 (1.99×) | — |
| `depthwise_conv2d` | common | — | 0.317 (18.30×) | — | 4.046 (177.20×) | — |
| `dot_product` | common | — | 0.007 (45.83×) | — | 2.997 (2.33×) | — |
| `elu` | common | — | 0.016 (486.43×) | — | 3.007 (1.99×) | — |
| `embedding` | common | — | 0.010 (18.49×) | — | 2.999 (2.33×) | — |
| `exp` | common | — | 0.016 (497.59×) | — | 3.006 (2.33×) | — |
| `frobenius_norm` | common | — | 0.076 (5.10×) | — | 3.059 (8.82×) | — |
| `gelu` | common | — | 0.016 (501.31×) | — | 3.003 (3.67×) | — |
| `hardsigmoid` | common | — | 0.016 (487.51×) | — | 3.002 (5.00×) | — |
| `hardswish` | common | — | 0.016 (485.73×) | — | 3.006 (2.31×) | — |
| `hardtanh` | common | — | 0.011 (18.97×) | — | 3.004 (2.29×) | — |
| `hinge_loss` | common | — | 0.018 (12.79×) | — | 0.023 (298.35×) | — |
| `huber_loss` | common | — | 0.161 (49.72×) | — | 7.015 (13.69×) | — |
| `kl_div_loss` | common | — | 0.156 (51.39×) | — | 7.042 (11.50×) | — |
| `l1_norm` | common | — | 0.064 (6.81×) | — | 20.964 (1.14×) | — |
| `l2_norm` | common | — | 0.063 (6.37×) | — | 20.005 (1.20×) | — |
| `layer_norm` | common | — | 0.059 (5.27×) | — | 6.977 (1.57×) | — |
| `leaky_relu` | common | — | 0.016 (502.02×) | — | 3.004 (5.94×) | — |
| `log` | common | — | 0.010 (21.45×) | — | 3.009 (4.99×) | — |
| `logsigmoid` | common | — | 0.018 (453.82×) | — | 3.007 (3.33×) | — |
| `logsumexp` | common | — | 0.071 (112.65×) | — | 32.003 (0.44×) | — |
| `manhattan_similarity` | common | — | 0.074 (6.52×) | — | 29.958 (1.07×) | — |
| `masked_cumsum` | common | — | 0.142 (3.30×) | — | 25.012 (1.28×) | — |
| `matmul_sub_mul_relu` | common | — | 0.075 (106.32×) | — | 0.000 (inf×) | — |
| `matrix_add` | common | — | 0.147 (2.50×) | — | 5.134 (4.67×) | — |
| `matrix_scale` | common | — | 0.083 (96.22×) | — | 3.048 (5.25×) | — |
| `matvec` | common | — | 0.059 (4.11×) | — | 10.005 (0.50×) | — |
| `max_pool1d` | common | — | 0.012 (47.30×) | — | 3.999 (5.99×) | — |
| `max_pool2d` | common | — | 0.103 (4.01×) | — | 6.995 (1.14×) | — |
| `max_pool3d` | common | — | 0.220 (5.00×) | — | 7.022 (1.14×) | — |
| `mish` | common | — | 0.011 (28.82×) | — | 3.009 (6.67×) | — |
| `mse_loss` | common | — | 0.234 (2.87×) | — | 3.176 (17.63×) | — |
| `outer_product` | common | — | 0.055 (5.66×) | — | 7.013 (1.14×) | — |
| `prelu` | common | — | 0.013 (18.05×) | — | 3.005 (5.32×) | — |
| `rect_mm` | common | — | 8.309 (0.94×) | — | 45.334 (0.53×) | — |
| `relu` | common | — | 0.015 (551.53×) | — | 3.002 (2.67×) | — |
| `rms_norm` | common | — | 0.055 (5.24×) | — | 3.038 (3.62×) | — |
| `rsqrt` | common | — | 0.011 (20.77×) | — | 3.010 (3.66×) | — |
| `selu` | common | — | 0.011 (20.97×) | — | 3.003 (2.33×) | — |
| `sigmoid` | common | — | 0.011 (21.11×) | — | 3.003 (2.32×) | — |
| `softmax` | common | — | 0.071 (4.11×) | — | 29.968 (0.53×) | — |
| `softplus` | common | — | 0.011 (26.28×) | — | 3.003 (4.00×) | — |
| `softsign` | common | — | 0.011 (19.26×) | — | 3.002 (5.00×) | — |
| `sqr_mm` | common | — | 1.129 (1.42×) | — | 21.026 (0.38×) | — |
| `swish` | common | — | 0.011 (21.67×) | — | 3.003 (2.00×) | — |
| `tanh` | common | — | 0.017 (448.92×) | — | 3.010 (2.66×) | — |
| `top_k` | common | — | 0.116 (8.47×) | — | 12.015 (3.99×) | — |
| `transpose_2d` | common | — | 0.177 (0.10×) | — | 17.043 (0.00×) | — |
| `triplet_margin_loss` | common | — | 0.134 (7.94×) | — | 32.011 (2.87×) | — |
| `variance` | common | — | 0.038 (13.68×) | — | 20.963 (1.14×) | — |
| `where` | common | — | 0.186 (2.46×) | — | 6.946 (3.45×) | — |
| `add_gelu` | standard | — | 0.007 (34.15×) | — | — | — |
| `add_norm` | standard | — | 0.138 (3.37×) | — | 12.016 (2.00×) | — |
| `add_relu` | standard | — | 0.006 (25.16×) | — | — | — |
| `add_silu` | standard | — | 0.006 (25.91×) | — | — | — |
| `add_swish` | standard | — | 0.006 (25.98×) | — | — | — |
| `attention_scores` | standard | — | 0.022 (9.58×) | — | 6.996 (1.57×) | — |
| `bias_add` | standard | — | 0.055 (5.00×) | — | 6.949 (3.45×) | — |
| `bias_gelu` | standard | — | 0.051 (7.92×) | — | 7.009 (5.85×) | — |
| `cross_entropy_loss` | standard | — | 0.090 (5.30×) | — | 23.954 (2.00×) | — |
| `dropout` | standard | — | 0.142 (3.13×) | — | 7.016 (3.42×) | — |
| `fused_add_rms_norm` | standard | — | 0.136 (3.28×) | — | 13.971 (2.07×) | — |
| `fused_qkv_projection` | standard | — | 0.031 (7.44×) | — | 3.010 (2.65×) | — |
| `gelu_linear` | standard | — | 1.148 (1.74×) | — | 18.029 (1.33×) | — |
| `gelu_residual` | standard | — | 0.007 (34.90×) | — | — | — |
| `group_norm` | standard | — | 0.023 (18.62×) | — | 7.015 (6.56×) | — |
| `instance_norm` | standard | — | 0.040 (10.77×) | — | 18.006 (2.61×) | — |
| `linear_bias` | standard | — | 0.031 (7.89×) | — | 3.016 (4.97×) | — |
| `llama_attention` | standard | — | 0.163 (2.33×) | — | 3.110 (12.54×) | — |
| `log_softmax` | standard | — | 0.084 (3.81×) | — | 31.016 (0.87×) | — |
| `log_softmax_cross_entropy` | standard | — | 0.094 (5.46×) | — | 42.007 (0.98×) | — |
| `masked_softmax` | standard | — | 0.139 (3.21×) | — | 33.997 (0.85×) | — |
| `matmul_gelu_softmax` | standard | — | 0.044 (7.17×) | — | 3.014 (4.67×) | — |
| `mul_gelu` | standard | — | 0.006 (32.16×) | — | — | — |
| `mul_relu` | standard | — | 0.006 (26.25×) | — | — | — |
| `mul_silu` | standard | — | 0.006 (29.34×) | — | — | — |
| `nll_loss` | standard | — | 0.069 (6.03×) | — | 28.993 (0.83×) | — |
| `residual_add` | standard | — | 0.140 (2.80×) | — | 6.998 (3.43×) | — |
| `residual_tanh` | standard | — | 0.006 (27.45×) | — | — | — |
| `rms_norm_linear` | standard | — | 1.364 (1.13×) | — | 21.008 (1.90×) | — |
| `rope_embedding` | standard | — | 0.005 (68.88×) | — | 2.997 (5.00×) | — |
| `scaled_dot_product` | standard | — | 0.153 (1.93×) | — | 0.060 (116.55×) | — |
| `silu_linear` | standard | — | 1.156 (6.92×) | — | 20.066 (1.20×) | — |
| `silu_residual` | standard | — | 0.007 (30.95×) | — | — | — |
| `softmax_attention` | standard | — | 0.033 (11.20×) | — | 12.002 (1.49×) | — |
| `swiglu` | standard | — | 0.052 (6.71×) | — | 9.993 (1.70×) | — |
| `alexnet` | full | — | 0.191 (1.57×) | — | 3.146 (7.63×) | — |
| `densenet` | full | — | 0.059 (4.63×) | — | 0.034 (202.74×) | — |
| `llama_decoder_layer` | full | — | 0.324 (1.30×) | — | 3.284 (10.35×) | — |
| `mha_attention` | full | — | 0.127 (1.96×) | — | — | — |
| `resnet` | full | — | 0.161 (2.12×) | — | 3.128 (7.67×) | — |
| `transformer_block` | full | — | 0.421 (0.90×) | — | 3.367 (8.89×) | — |

_116 kernels total. Chips covered: M2 (116), M4 (105)._
