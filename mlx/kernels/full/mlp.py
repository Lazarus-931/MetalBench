import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """MLP block: x @ W1 → GELU → @ W2 → GELU → @ W3.
    Shapes: x (16, 128), W1 (128, 512), W2 (512, 128), W3 (128, 10). Out (16, 10)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, W1, W2, W3):
        h = nn.gelu(x @ W1)
        h = nn.gelu(h @ W2)
        return h @ W3
