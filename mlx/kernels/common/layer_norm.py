import mlx.core as mx
from mlx import nn
from mlx_helpers import Output, Scalar


class Model(nn.Module):
    """Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta."""
    def __init__(self, dims: int = 1024, eps: float = 1e-5):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(dims, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        return self.ln(x)


N, eps = 1024, 1e-5

metal_function = "layer_norm_f32"
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
    return [
        Scalar(binding=2, dtype="u32", value=N),
        Scalar(binding=3, dtype="f32", value=eps),
    ]


def grid(inputs):
    return (N, N, 1)


def flops(inputs):
    return N * N * 7


def bytes(inputs):
    return 2 * N * N * 4
