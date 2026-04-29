import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Rectangular matrix multiplication: C = A @ B  (MxK @ KxN -> MxN)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """Returns C = A @ B of shape (M, N)."""
        return mx.matmul(A, B)


M = 1024
K = 4096
N = 2048


def get_inputs():
    A = mx.random.normal((M, K), dtype=mx.float32)
    B = mx.random.normal((K, N), dtype=mx.float32)
    return [A, B]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()


def reference(a, b):
    return _model.forward(a, b)
