import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Max pool 1D, NLC. Kernel=3, stride=2."""
    def __init__(self, k: int = 3, s: int = 2):
        super(Model, self).__init__()
        self.k, self.s = k, s

    def forward(self, x: mx.array) -> mx.array:
        # x: (N, L, C). Slide window of size k with stride s, take max.
        N, L, C = x.shape
        out_L = (L - self.k) // self.s + 1
        cols = [x[:, i*self.s : i*self.s + self.k, :] for i in range(out_L)]
        stacked = mx.stack(cols, axis=1)            # (N, out_L, k, C)
        return mx.max(stacked, axis=2)              # (N, out_L, C)
