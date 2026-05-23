import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: conv3d → softmax along channel axis → maxpool 2x2x2 → maxpool 2x2x2.
    x (4, 32, 32, 32, 32), w (64, 3, 3, 3, 32) → conv → (4, 30, 30, 30, 64)
    → softmax over last dim → maxpool stride 2 → (4, 15, 15, 15, 64)
    → maxpool stride 2 → (4, 7, 7, 7, 64) (drops the odd row/col)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w):
        y = mx.conv3d(x, w, stride=1, padding=0)
        y = mx.softmax(y, axis=-1)
        # Maxpool 2x2x2 stride 2 — reshape-then-max
        N, D, H, W, C = y.shape
        D2, H2, W2 = D // 2, H // 2, W // 2
        y = y[:, :D2*2, :H2*2, :W2*2, :].reshape(N, D2, 2, H2, 2, W2, 2, C).max(axis=(2, 4, 6))
        # Second maxpool
        N, D, H, W, C = y.shape
        D2, H2, W2 = D // 2, H // 2, W // 2
        y = y[:, :D2*2, :H2*2, :W2*2, :].reshape(N, D2, 2, H2, 2, W2, 2, C).max(axis=(2, 4, 6))
        return y
