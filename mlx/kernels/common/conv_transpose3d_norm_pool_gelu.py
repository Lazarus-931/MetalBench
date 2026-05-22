import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: conv_transpose3d → +sum (per-channel) → LayerNorm → AvgPool2x2x2 → GELU.
    x (2, 8, 8, 8, 16) → conv_transpose3d (32, 3, 3, 3, 16) stride=1 → (2, 10, 10, 10, 32)
    → + sum_term (32,) → LayerNorm over last dim → avgpool 2x2x2 stride=2 → (2, 5, 5, 5, 32)
    → GELU."""
    def __init__(self, eps: float = 1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x, w, sum_term):
        y = mx.conv_transpose3d(x, w, stride=1, padding=0)
        y = y + sum_term
        mean = mx.mean(y, axis=-1, keepdims=True)
        var = mx.var(y, axis=-1, keepdims=True, ddof=0)
        y = (y - mean) * mx.rsqrt(var + self.eps)
        N, D, H, W, C = y.shape
        D2, H2, W2 = D // 2, H // 2, W // 2
        y = y[:, :D2*2, :H2*2, :W2*2, :].reshape(N, D2, 2, H2, 2, W2, 2, C).mean(axis=(2, 4, 6))
        return nn.gelu(y)
