import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Hinge loss element-wise: max(0, 1 - pred*target). target in {-1, +1} encoded as f32."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pred: mx.array, target: mx.array) -> mx.array:
        return mx.maximum(1.0 - pred * target, 0.0)
