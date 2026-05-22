import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """AlexNet-mini forward pass. NHWC throughout. Input (1, 32, 32, 3) → (1, 10).

    Pipeline:
      conv1 (k=5, no pad)    32→28×28×32
      maxpool 2×2 s=2        14×14×32
      ReLU
      conv2 (k=3, no pad)    12×12×64
      maxpool 2×2 s=2        6×6×64
      ReLU
      conv3 (k=3, no pad)    4×4×128
      maxpool 2×2 s=2        2×2×128
      ReLU
      flatten → 512
      fc1 → 256, ReLU
      fc2 → 10
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, W_c1, W_c2, W_c3, W_fc1, W_fc2):
        y = mx.conv2d(x, W_c1, stride=1, padding=0)
        y = self._maxpool2(y)
        y = mx.maximum(y, 0)
        y = mx.conv2d(y, W_c2, stride=1, padding=0)
        y = self._maxpool2(y)
        y = mx.maximum(y, 0)
        y = mx.conv2d(y, W_c3, stride=1, padding=0)
        y = self._maxpool2(y)
        y = mx.maximum(y, 0)
        N, H, W, C = y.shape
        y = y.reshape(N, H * W * C)
        y = mx.maximum(y @ W_fc1, 0)
        return y @ W_fc2

    def _maxpool2(self, x):
        N, H, W, C = x.shape
        return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))
