import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """conv3d → ReLU → LeakyReLU(0.01) → GELU → Sigmoid → +bias (per-channel).
    x (4, 32, 32, 32, 32), w (64, 3, 3, 3, 32), b (64,) → (4, 30, 30, 30, 64)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w, b):
        y = mx.conv3d(x, w, stride=1, padding=0)
        y = mx.maximum(y, 0.0)
        y = mx.where(y > 0, y, 0.01 * y)
        y = nn.gelu(y)
        y = mx.sigmoid(y)
        return y + b
