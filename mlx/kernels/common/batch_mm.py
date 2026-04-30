import mlx.core as mx
from mlx import nn
from mlx_helpers import Output, Scalar, batched_matmul_spec


class Model(nn.Module):
    """Batched matrix multiplication: C[b] = A[b] @ B[b]."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        return mx.matmul(A, B)


batch, M, K, N = 128, 128, 256, 512
globals().update(batched_matmul_spec("batch_matmul_f32", batch, M, N, K, BK=16))


def get_inputs():
    A = mx.random.normal((batch, M, K), dtype=mx.float32)
    B = mx.random.normal((batch, K, N), dtype=mx.float32)
    return [A, B]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()
def reference(a, b):
    return _model.forward(a, b)
