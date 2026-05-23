import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Max pool 2D, NHWC. Kernel=2x2, stride=2. Assumes H, W divisible by stride."""
    def __init__(self, k: int = 2, s: int = 2):
        super(Model, self).__init__()
        self.k, self.s = k, s

    def forward(self, x: mx.array) -> mx.array:
        N, H, W, C = x.shape
        k = self.k
        # Reshape into (N, H/k, k, W/k, k, C) and max-reduce the two k-axes.
        x2 = x.reshape(N, H // k, k, W // k, k, C)
        return x2.max(axis=(2, 4))
