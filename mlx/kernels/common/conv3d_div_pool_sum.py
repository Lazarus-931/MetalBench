import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Fused: conv3d → divide → max over channels → global avg pool over D,H,W → +b → sum.
    x (4, 32, 32, 32, 32), w (64, 3, 3, 3, 32), b scalar=0.5 → output scalar(1,)."""
    def __init__(self, div_val: float = 2.0, bias_val: float = 0.5):
        super(Model, self).__init__()
        self.div_val = div_val
        self.bias_val = bias_val

    def forward(self, x, w):
        y = mx.conv3d(x, w, stride=1, padding=0)         # (4, 30, 30, 30, 64)
        y = y / self.div_val
        y = mx.max(y, axis=-1)                            # max over K → (4, 30, 30, 30)
        y = mx.mean(y, axis=(1, 2, 3), keepdims=False)    # global avg → (4,)
        y = y + self.bias_val
        return mx.sum(y, keepdims=True).reshape(1)        # scalar
