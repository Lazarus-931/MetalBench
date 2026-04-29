import mlx
import mlx.core as mx
from mlx import nn

from mlx_helpers import Output, Scalar


class Model(nn.Module):
    """
    Single square matrix multiplication: C = A @ B  (N × N · N × N).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """
        Args:
            A: Input array of shape (N, N).
            B: Input array of shape (N, N).

        Returns:
            Output array of shape (N, N).
        """
        return mx.matmul(A, B)


N = 1024


def get_inputs():
    A = mx.random.normal((N, N), dtype=mx.float32)
    B = mx.random.normal((N, N), dtype=mx.float32)
    return [A, B]


def get_init_inputs():
    return []





BM = BN = 64
TG_THREADS = 256

assert N % BM == 0, "sqr_mm requires N % 64 == 0"

metal_function = "sqr_matmul_f32"
threadgroup    = (TG_THREADS, 1, 1)
input_bindings = (0, 1)
outputs        = [Output(binding=2, dtype="f32", shape=(N, N))]
rtol, atol     = 1e-3, 1e-3


BEST_FOR = ["all"]


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()
def reference(a, b):
    return _model.forward(a, b)


def scalars(inputs):
    return [Scalar(binding=3, dtype="u32", value=N)]


def grid(inputs):
    return ((N // BM) * TG_THREADS, N // BN, 1)


def flops(inputs):
    # 2 FLOPs per multiply-accumulate, N^3 MACs for an N×N matmul.
    return 2 * N * N * N


def bytes(inputs):
    # Read A (N²) + read B (N²) + write C (N²), all f32.
    return 3 * N * N * 4
