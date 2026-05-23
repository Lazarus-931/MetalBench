import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """InstanceNorm: μ,σ per-sample per-channel over spatial dims (H,W)."""
    def __init__(self, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        # x shape: (N, C, H, W). Reduce over (H, W).
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=(2, 3), keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps)
