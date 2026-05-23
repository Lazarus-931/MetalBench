import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Element-wise natural log: out = log(|x| + tiny).
    abs + tiny added so random-normal test inputs don't produce NaN/-inf."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.log(mx.abs(x) + 1e-30)
