# MetalBench Best Times

| kernel | type | apple-m2 Metal | apple-m2 speedup |
|---||---||---||---|
| add_norm | ? | 0.187ms | 3.1× |
| batch_mm | matmul | 2.693ms | 0.9× |
| cosine_similarity | reduce | 0.180ms | 4.2× |
| cumprod | scan | 0.127ms | 3.1× |
| cumsum | scan | 0.126ms | 3.0× |
| cumsum_reverse | scan | 0.127ms | 3.9× |
| dot_product | reduce | 0.014ms | 15.8× |
| gelu | elem | 0.020ms | 12.8× |
| hardsigmoid | elem | 0.020ms | 14.8× |
| hardswish | elem | 0.020ms | 15.1× |
| l1_norm | reduce | 0.115ms | 2.9× |
| l2_norm | reduce | 0.125ms | 2.8× |
| layer_norm | reduce | 0.191ms | 2.1× |
| leaky_relu | elem | 0.020ms | 12.9× |
| logsigmoid | elem | 0.021ms | 13.5× |
| manhattan_similarity | reduce | 0.123ms | 4.0× |
| matrix_add | elem | 0.165ms | 2.1× |
| matrix_scale | elem | 0.084ms | 3.3× |
| matvec | matmul | 0.122ms | 2.2× |
| mse_loss | reduce | 0.534ms | 1.9× |
| outer_product | matmul | 0.404ms | 0.9× |
| rect_mm | matmul | 9.560ms | 0.8× |
| relu | elem | 0.020ms | 10.0× |
| rms_norm | reduce | 0.137ms | 2.3× |
| selu | elem | 0.020ms | 12.1× |
| sigmoid | elem | 0.020ms | 12.2× |
| silu_linear | ? | 1.643ms | 1.1× |
| softmax | reduce | 0.212ms | 1.5× |
| sqr_mm | matmul | 1.131ms | 1.2× |
| swish | elem | 0.020ms | 12.6× |
| tanh | elem | 0.020ms | 13.4× |
| trace | reduce | 0.010ms | 17.6× |
| transpose_2d | ? | 0.225ms | 0.1× |

---
**Chips:** apple-m2

To submit: fork, edit `.metal`, `./bench <name>`, update PR.
