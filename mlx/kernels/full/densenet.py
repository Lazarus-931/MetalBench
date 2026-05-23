import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """DenseNet-mini: stem + 2 dense layers (with channel concat) + GAP + FC.

    NHWC, padding=1 throughout so spatial dims stay (16, 16). growth_rate=12.

    Input  (1, 16, 16, 3)
      stem  W_stem (12, 3, 3, 3)   pad=1  → ReLU → h0 (1, 16, 16, 12)
      d1    W_d1   (12, 3, 3, 12)  pad=1  → ReLU → c1 (1, 16, 16, 12)
            concat([h0, c1], axis=C)             → h1 (1, 16, 16, 24)
      d2    W_d2   (12, 3, 3, 24)  pad=1  → ReLU → c2 (1, 16, 16, 12)
            concat([h1, c2], axis=C)             → h2 (1, 16, 16, 36)
      global avg pool over (H, W)                → (1, 36)
      fc    W_fc   (36, 10)                       → (1, 10)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, W_stem, W_d1, W_d2, W_fc):
        h0 = mx.maximum(mx.conv2d(x,  W_stem, stride=1, padding=1), 0)
        c1 = mx.maximum(mx.conv2d(h0, W_d1,   stride=1, padding=1), 0)
        h1 = mx.concatenate([h0, c1], axis=3)
        c2 = mx.maximum(mx.conv2d(h1, W_d2,   stride=1, padding=1), 0)
        h2 = mx.concatenate([h1, c2], axis=3)
        g  = mx.mean(h2, axis=(1, 2))
        return g @ W_fc
