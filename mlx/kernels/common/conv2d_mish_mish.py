import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused conv2d → Mish → Mish. NHWC.
    x (8, 64, 64, 64) @ w (128, 3, 3, 64) → (8, 62, 62, 128). mish(z) = z*tanh(softplus(z))."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w):
        def mish(z):
            return z * mx.tanh(mx.logaddexp(z, mx.zeros_like(z)))
        return mish(mish(mx.conv2d(x, w, stride=1, padding=0)))
