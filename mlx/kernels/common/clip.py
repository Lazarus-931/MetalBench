import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Element-wise clamp to [lo, hi]. Defaults match the Metal kernel scalars."""
    def __init__(self, lo: float = -1.0, hi: float = 1.0):
        super(Model, self).__init__()
        self.lo = lo
        self.hi = hi

    def forward(self, x: mx.array) -> mx.array:
        return mx.clip(x, self.lo, self.hi)
