import mlx.core as mx
from mlx import nn
from mlx_helpers import Output, Scalar


class Model(nn.Module):
    """L1 norm: sum of absolute values."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Returns L1 norm over the last dimension."""
        return mx.sum(mx.abs(x), axis=-1)


N = 1024

metal_function = "l1_norm_f32"
threadgroup    = (1024, 1, 1)
input_bindings = (0,)
outputs        = [Output(binding=1, dtype="f32", shape=(N,))]
rtol, atol     = 1e-3, 1e-3


def get_inputs():
    x = mx.random.normal((N, N), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()
def reference(x):
    return _model.forward(x)


def scalars(inputs):
    return [Scalar(binding=2, dtype="u32", value=N)]


def grid(inputs):
    return (N, N, 1)


def flops(inputs):
    return N * N * 2


def bytes(inputs):
    return N * N * 4 + N * 4
