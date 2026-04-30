import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """LogSigmoid activation: log(sigmoid(x))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Returns log-sigmoid activation element-wise."""
        return nn.log_sigmoid(x)


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
