import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused conv2d + ReLU + per-channel bias. NHWC.

    Input  x (8, 64, 64, 64), weight w (128, 3, 3, 64), bias b (128,).
    Output (8, 62, 62, 128) = ReLU(conv2d(x, w) + b).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w, b):
        y = mx.conv2d(x, w, stride=1, padding=0, dilation=1, groups=1)
        return mx.maximum(y + b, 0.0)
