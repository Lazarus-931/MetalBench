import mlx.core as mx
from mlx import nn
from mlx_helpers import Output, Scalar, matmul_spec


class Model(nn.Module):
    """Matrix multiplication."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        return mx.matmul(A, B)


M, N, K = 1024, 1024, 1024
globals().update(matmul_spec("sqr_matmul_f32", M, N, K))


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
