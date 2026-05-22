# apple-m4 Results

| kernel | time (ms) | speedup | GFLOPS | GB/s |
|---|---|---|---|---|
| alexnet | 0.149 | 56.67× | 74 | 5 |
| avg_pool2d | 0.110 | 9.33× | 23 | 93 |
| avg_pool3d | 0.193 | 5.42× | 24 | 97 |
| batch_mm | 2.393 | 0.70× | 1785 | 48 |
| cosine_similarity | 0.078 | 12.59× | 65 | 104 |
| cumprod | 0.073 | 2.81× | 10 | 81 |
| cumsum | 0.063 | 4.12× | 13 | 104 |
| cumsum_reverse | 0.069 | 5.52× | 24 | 99 |
| dot_product | 0.007 | 25.11× | 4 | 17 |
| frobenius_norm | 0.067 | 4.25× | 30 | 60 |
| fused_qkv_projection | 0.034 | 26.67× | 662 | 19 |
| gelu | 0.014 | 13.17× | 92 | 148 |
| gelu_linear | 0.827 | 2.52× | 2567 | 15 |
| hardsigmoid | 0.014 | 17.81× | 56 | 151 |
| hardswish | 0.012 | 20.22× | 89 | 178 |
| l1_norm | 0.039 | 9.15× | 53 | 107 |
| l2_norm | 0.040 | 8.91× | 77 | 104 |
| layer_norm | 0.058 | 118.70× | 78 | 125 |
| leaky_relu | 0.014 | 18.36× | 56 | 149 |
| llama_decoder_layer | 0.648 | 1.62× | 16 | 0 |
| logsigmoid | 0.014 | 15.50× | 90 | 145 |
| manhattan_similarity | 0.079 | 6.54× | 25 | 101 |
| matrix_add | 0.117 | 3.10× | 8 | 104 |
| matrix_scale | 0.064 | 4.36× | 14 | 118 |
| matvec | 0.037 | 25.63× | 53 | 107 |
| mlp | 0.031 | 170.56× | 81 | 10 |
| mse_loss | 0.409 | 1.79× | 7 | 30 |
| outer_product | 0.151 | 1.89× | 13 | 54 |
| rect_mm | 6.232 | 0.89× | 2720 | 9 |
| relu | 0.014 | 15.63× | 18 | 151 |
| resnet | 0.184 | 38.79× | 49 | 0 |
| rms_norm | 0.083 | 3.64× | 61 | 98 |
| rms_norm_linear | 0.856 | 1.47× | 2370 | 13 |
| selu | 0.014 | 13.60× | 93 | 149 |
| sigmoid | 0.014 | 17.02× | 74 | 149 |
| softmax | 0.086 | 3.31× | 60 | 97 |
| sqr_mm | 0.870 | 1.29× | 2262 | 13 |
| swish | 0.014 | 16.41× | 93 | 149 |
| tanh | 0.014 | 16.00× | 75 | 151 |
| trace | 0.005 | 40.33× | 0 | 1 |
| transformer_block | 0.343 | 21.89× | 48 | 1 |
| transpose_2d | 0.168 | 0.11× | 0 | 94 |

_42 kernels._

