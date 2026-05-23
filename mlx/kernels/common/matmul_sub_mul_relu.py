import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: y = ReLU((x @ w - sub_val) * mul_val).
    x (256, 256), w (256, 256), sub_val 0.5, mul_val 2.0."""
    def __init__(self, sub_val: float = 0.5, mul_val: float = 2.0):
        super(Model, self).__init__()
        self.sub_val = sub_val
        self.mul_val = mul_val

    def forward(self, x, w):
        return mx.maximum((x @ w - self.sub_val) * self.mul_val, 0.0)
