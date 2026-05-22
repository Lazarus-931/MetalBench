import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Average pool 1D, NLC. Kernel=3, stride=2."""
    def __init__(self, k: int = 3, s: int = 2):
        super(Model, self).__init__()
        self.k, self.s = k, s

    def forward(self, x: mx.array) -> mx.array:
        N, L, C = x.shape
        out_L = (L - self.k) // self.s + 1
        cols = [x[:, i*self.s : i*self.s + self.k, :] for i in range(out_L)]
        return mx.mean(mx.stack(cols, axis=1), axis=2)
