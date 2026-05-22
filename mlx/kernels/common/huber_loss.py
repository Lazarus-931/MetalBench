import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Huber loss per element: 0.5·r² if |r|≤δ else δ·(|r|-0.5δ). r = pred - target. δ=1."""
    def __init__(self, delta: float = 1.0):
        super(Model, self).__init__()
        self.delta = delta

    def forward(self, pred: mx.array, target: mx.array) -> mx.array:
        r = pred - target
        absr = mx.abs(r)
        quad = 0.5 * r * r
        lin = self.delta * (absr - 0.5 * self.delta)
        return mx.where(absr <= self.delta, quad, lin)
