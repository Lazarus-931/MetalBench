import mlx.core as mx

from mlx_helpers import Output, Scalar, element_wise_spec


class Model(nn.Module):
    """Element-wise sigmoid activation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.sigmoid(x)


N_el = 16 * 16384
globals().update(element_wise_spec("sigmoid_f32", N_el, flops_mul=4))


def get_inputs():
    x = mx.random.normal((16, 16384), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []


def make_inputs(seed: int):
    mx.random.seed(seed)
    return get_inputs()


_model = Model()
def reference(x):
    return _model.forward(x)
