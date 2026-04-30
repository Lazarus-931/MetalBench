import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """BatchNorm: normalize each feature over the batch dimension."""
    def __init__(self, num_features: int = 256, eps: float = 1e-5):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm(num_features, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        """Returns batch-normalized output of same shape."""
        return self.bn(x)


def get_inputs():
    x = mx.random.normal((1024, 256), dtype=mx.float32)
    return [x]


def get_init_inputs():
    return []
