import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Attention scores only (no V): softmax(Q @ K^T / √d). Output (S, S)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q: mx.array, k: mx.array) -> mx.array:
        d = q.shape[-1]
        scores = (q @ k.T) / mx.sqrt(mx.array(float(d)))
        return mx.softmax(scores, axis=-1)
