import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """BatchNorm: normalize each feature over the batch dimension (1024, 256)."""
    def __init__(self, num_features: int = 256, eps: float = 1e-5):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm(num_features, eps=eps)

    def forward(self, x: mx.array) -> mx.array:
        return self.bn(x)
