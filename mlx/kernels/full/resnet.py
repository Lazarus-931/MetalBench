import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """ResNet-mini: stem conv + 1 residual block + GAP + FC.

    NHWC, padding=1 throughout so spatial dims stay (32, 32) until pooling.
    Input  (1, 32, 32, 3)
      stem  (16, 3, 3, 3)  pad=1  → (1, 32, 32, 16)
      ReLU
      block: conv (16, 3, 3, 16) pad=1 → ReLU → conv (16, 3, 3, 16) pad=1 → +input → ReLU
      global avg pool → (1, 16)
      fc → (1, 10)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, W_stem, W_a, W_b, W_fc):
        h = mx.conv2d(x, W_stem, stride=1, padding=1)
        h = mx.maximum(h, 0)
        residual = h
        h = mx.conv2d(h, W_a, stride=1, padding=1)
        h = mx.maximum(h, 0)
        h = mx.conv2d(h, W_b, stride=1, padding=1)
        h = h + residual
        h = mx.maximum(h, 0)
        h = mx.mean(h, axis=(1, 2))
        return h @ W_fc
