import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Element-wise reciprocal square root: out = 1 / sqrt(|x|).
    abs is applied so random-normal test inputs don't produce NaN."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.rsqrt(mx.abs(x))
