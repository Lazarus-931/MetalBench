import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Average pool 3D, NDHWC. Kernel=2x2x2, stride=2."""
    def __init__(self, k: int = 2, s: int = 2):
        super(Model, self).__init__()
        self.k, self.s = k, s

    def forward(self, x: mx.array) -> mx.array:
        N, D, H, W, C = x.shape
        k = self.k
        x2 = x.reshape(N, D // k, k, H // k, k, W // k, k, C)
        return x2.mean(axis=(2, 4, 6))
