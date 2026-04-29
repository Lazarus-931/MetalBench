import mlx
import mlx.core as mx
from mlx import nn

from mlx_helpers import Output, Scalar


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
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


BM = 64
BN = 64
BK = 16
TG_THREADS = 256

assert M % BM == 0, "rect_mm requires M % 64 == 0"
assert N % BN == 0, "rect_mm requires N % 64 == 0"
assert K % BK == 0, "rect_mm requires K % 16 == 0"

metal_function = "rect_matmul_f32"
threadgroup    = (TG_THREADS, 1, 1)
input_bindings = (0, 1)
outputs        = [Output(binding=2, dtype="f32", shape=(M, N))]
rtol, atol     = 1e-3, 1e-3

BEST_FOR = ["all"]


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()
def reference(a, b):
    return _model.forward(a, b)


def scalars(inputs):
    return [
        Scalar(binding=3, dtype="u32", value=M),
        Scalar(binding=4, dtype="u32", value=N),
        Scalar(binding=5, dtype="u32", value=K),
    ]


def grid(inputs):
    return ((N // BN) * TG_THREADS, M // BM, 1)


def flops(inputs):
    return 2 * M * N * K


def bytes(inputs):
    return 4 * (M * K + K * N + M * N)
