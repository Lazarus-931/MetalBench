import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: y = tanh(conv_transpose2d(x, w) - sub_val).
    x (8, 32, 32, 64), w (128, 3, 3, 64) → (8, 65, 65, 128). stride=2, sub_val=0.5."""
    def __init__(self, sub_val: float = 0.5):
        super(Model, self).__init__()
        self.sub_val = sub_val

    def forward(self, x, w):
        return mx.tanh(mx.conv_transpose2d(x, w, stride=2, padding=0) - self.sub_val)
