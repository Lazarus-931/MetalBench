import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused QKV projection: x @ W_qkv → concatenated (Q | K | V) along last dim.
    Input x: (S, D_in). W_qkv: (D_in, 3 * D_head). Output: (S, 3 * D_head).
    Downstream code slices into Q, K, V."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, w_qkv: mx.array) -> mx.array:
        return x @ w_qkv
