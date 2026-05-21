import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Stable log-softmax per row: log(softmax(x)) = x − logsumexp(x)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return x - mx.logsumexp(x, axis=-1, keepdims=True)
