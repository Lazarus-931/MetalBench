import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Reverse cumulative sum along last dim: out[i] = sum(x[i..N-1])."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        rev = x[..., ::-1]
        return mx.cumsum(rev, axis=-1)[..., ::-1]
