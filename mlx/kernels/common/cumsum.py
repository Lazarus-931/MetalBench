import mlx.core as mx
from mlx import nn
from mlx_helpers import Output, Scalar


class Model(nn.Module):
    """Cumulative sum: out[i] = sum(x[0..i]) along the last dim."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        """Returns cumulative sum along the last dimension."""
        return mx.cumsum(x, axis=-1)


N = 1024

metal_function = "cumsum_f32"
threadgroup    = (1024, 1, 1)
input_bindings = (0,)
outputs        = [Output(binding=1, dtype="f32", shape=(N, N))]
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
    return N * N


def bytes(inputs):
    return 2 * N * N * 4
