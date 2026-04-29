import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Square matrix multiplication: C = A @ B  (N x N . N x N)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """Returns C = A @ B of shape (N, N)."""
        return mx.matmul(A, B)


N = 1024


def get_inputs():
    A = mx.random.normal((N, N), dtype=mx.float32)
    B = mx.random.normal((N, N), dtype=mx.float32)
    return [A, B]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()


def reference(a, b):
    return _model.forward(a, b)
