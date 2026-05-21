import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Per-row variance: Var(x) = mean(x²) − mean(x)². Population (ddof=0)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.var(x, axis=-1, ddof=0)
