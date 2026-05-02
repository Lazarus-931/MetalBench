# MetalBench Best Times

| kernel | type | M2 Metal | M2 speedup | M4 Metal | M4 speedup |
|---|---|---|---|---|---|
| batch_mm | matmul | 2.693ms | 0.9× | 2.406ms | 0.7× |
| cosine_similarity | reduce | 0.184ms | 4.5× | 0.101ms | 8.2× |
| cumprod | scan | 0.131ms | 3.4× | 0.103ms | 2.8× |
| cumsum | scan | 0.129ms | 3.0× | 0.081ms | 4.1× |
| cumsum_reverse | scan | 0.129ms | 3.8× | 0.084ms | 5.5× |
| dot_product | reduce | 0.014ms | 15.8× | 0.007ms | 25.1× |
| gelu | elem | 0.020ms | 12.8× | 0.014ms | 13.2× |
| hardsigmoid | elem | 0.020ms | 14.8× | 0.014ms | 17.8× |
| hardswish | elem | 0.020ms | 15.1× | 0.012ms | 20.2× |
| l1_norm | reduce | 0.116ms | 3.8× | 0.039ms | 9.1× |
| l2_norm | reduce | 0.125ms | 2.8× | 0.040ms | 8.9× |
| layer_norm | reduce | 0.191ms | 2.1× | 0.085ms | 3.8× |
| leaky_relu | elem | 0.020ms | 13.6× | 0.014ms | 18.4× |
| logsigmoid | elem | 0.021ms | 13.5× | 0.014ms | 15.5× |
| manhattan_similarity | reduce | 0.134ms | 3.7× | 0.083ms | 6.5× |
| matrix_add | elem | 0.169ms | 2.3× | 0.121ms | 3.1× |
| matrix_scale | elem | 0.099ms | 2.9× | 0.071ms | 4.4× |
| matvec | matmul | 0.122ms | 2.2× | 0.042ms | 5.6× |
| mse_loss | reduce | 0.534ms | 1.9× | 0.417ms | 1.8× |
| outer_product | matmul | 0.405ms | 0.9× | 0.154ms | 1.9× |
| rect_mm | matmul | 9.560ms | 0.8× | 6.316ms | 0.9× |
| relu | elem | 0.020ms | 10.0× | 0.014ms | 15.6× |
| rms_norm | reduce | 0.138ms | 2.4× | 0.085ms | 3.6× |
| selu | elem | 0.020ms | 12.1× | 0.014ms | 13.6× |
| sigmoid | elem | 0.020ms | 12.2× | 0.014ms | 17.0× |
| softmax | reduce | 0.212ms | 1.5× | -- | -- |
| sqr_mm | matmul | 1.131ms | 1.2× | 0.949ms | 1.3× |
| swish | elem | 0.020ms | 12.6× | 0.014ms | 16.4× |
| tanh | elem | 0.020ms | 13.2× | 0.014ms | 16.0× |
| trace | reduce | 0.010ms | 17.6× | 0.005ms | 40.3× |
| transpose_2d | copy | 0.225ms | 0.1× | 0.641ms | 0.0× |

---

**Chips:** Apple M2 (8 CPU / 8 GPU / 9 GB) — Apple M4 (10 CPU / 10 GPU / 17 GB)

To submit a faster kernel: fork, edit the `.metal` file, run `./bench <name>`,
update this file with your new best time, and open a PR.