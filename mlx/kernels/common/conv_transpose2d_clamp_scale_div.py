import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: y = clamp(clamp(conv_transpose2d(x, w) + b, -1, 1) * scale, -1, 1) / div.
    x (8, 32, 32, 64), w (128, 3, 3, 64) → (8, 65, 65, 128). b (128,)."""
    def __init__(self, scale: float = 2.0, div: float = 2.0):
        super(Model, self).__init__()
        self.scale = scale
        self.div = div

    def forward(self, x, w, b):
        y = mx.conv_transpose2d(x, w, stride=2, padding=0) + b
        y = mx.clip(y, -1.0, 1.0) * self.scale
        return mx.clip(y, -1.0, 1.0) / self.div
