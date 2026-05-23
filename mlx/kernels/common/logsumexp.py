import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Numerically stable log(sum(exp(x))) along the last dim.
    Returns shape (R,) for input (R, C). Foundation for softmax / cross_entropy."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.logsumexp(x, axis=-1)
