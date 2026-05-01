# MetalBench Best Times

**Device:** Apple M2

| kernel | type | Metal (ms) | MLX (ms) | speedup | GFLOPS | GB/s |
|---|---|---|---|---|---|---|
| dot_product | reduction | 0.014 | 0.227 | 15.8× | 2 | 9.1 |
| hardsigmoid | element-wise | 0.020 | 0.292 | 14.8× | 40 | 106.4 |
| leaky_relu | element-wise | 0.020 | 0.275 | 13.6× | 40 | 106.0 |
| logsigmoid | element-wise | 0.021 | 0.296 | 13.5× | 63 | 101.3 |
| gelu | element-wise | 0.020 | 0.290 | 12.8× | 66 | 105.1 |
| swish | element-wise | 0.020 | 0.250 | 12.6× | 66 | 105.5 |
| sigmoid | element-wise | 0.020 | 0.243 | 12.2× | 53 | 105.5 |
| selu | element-wise | 0.020 | 0.239 | 12.1× | 66 | 106.2 |
| relu | element-wise | 0.020 | 0.221 | 10.0× | 13 | 107.3 |
| l1_norm | reduction | 0.116 | 0.385 | 3.8× | 18 | 36.3 |
| cumsum_reverse | scan | 0.129 | 0.519 | 3.8× | 16 | 64.9 |
| cumprod | scan | 0.131 | 0.450 | 3.4× | 8 | 64.1 |
| cumsum | scan | 0.129 | 0.391 | 3.0× | 8 | 65.1 |
| l2_norm | reduction | 0.125 | 0.377 | 2.8× | 25 | 33.7 |
| rms_norm | reduction | 0.138 | 0.331 | 2.4× | 38 | 60.6 |
| matrix_add | element-wise | 0.169 | 0.369 | 2.3× | 6 | 74.5 |
| matvec | matmul | 0.122 | 0.240 | 2.2× | 17 | 34.5 |
| layer_norm | reduction | 0.191 | 0.380 | 2.1× | 39 | 44.0 |
| mse_loss | other | 0.534 | 1.002 | 1.9× | 6 | 23.5 |
| sqr_mm | matmul | 1.131 | 1.192 | 1.2× | 1899 | 11.1 |
| outer_product | matmul | 0.405 | 0.384 | 0.9× | 5 | 20.8 |
| batch_mm | matmul | 2.693 | 2.147 | 0.9× | 1595 | 43.6 |
| rect_mm | matmul | 9.560 | 8.128 | 0.8× | 1797 | 6.1 |
| transpose_2d | other | 0.225 | 0.018 | 0.1× | -- | 74.5 |

---

To submit a faster kernel: fork, edit the `.metal` file, run `./bench <name>`,
update this file with your new best time, and open a PR.