import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused y = softmax(GELU(x @ w), axis=-1).
    x (256, 256), w (256, 256), output (256, 256). Per-row softmax."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w):
        return mx.softmax(nn.gelu(x @ w), axis=-1)
