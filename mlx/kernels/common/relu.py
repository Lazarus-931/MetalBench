import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """ReLU activation: out = max(x, 0)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Applies ReLU activation element-wise to input array of any shape."""
        return mx.maximum(x, 0.0)


batch_size = 16
dim = 16384


def get_inputs():
    x = mx.random.normal((batch_size, dim), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()


def reference(x):
    return _model.forward(x)
