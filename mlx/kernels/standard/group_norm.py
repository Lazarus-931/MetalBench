import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """GroupNorm: μ,σ per group of channels over (group_channels, H, W).
    Shape (8, 64, 32, 32) with groups=8 → 8 channels per group."""
    def __init__(self, groups: int = 8, eps: float = 1e-5):
        super(Model, self).__init__()
        self.groups = groups
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        N, C, H, W = x.shape
        g = self.groups
        x_g = x.reshape(N, g, C // g, H, W)
        mean = mx.mean(x_g, axis=(2, 3, 4), keepdims=True)
        var = mx.mean((x_g - mean) ** 2, axis=(2, 3, 4), keepdims=True)
        y = (x_g - mean) * mx.rsqrt(var + self.eps)
        return y.reshape(N, C, H, W)
