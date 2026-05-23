import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """NLL loss: per-row -sum(y_onehot * log_probs). Returns (N,)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, log_probs: mx.array, y_onehot: mx.array) -> mx.array:
        return -mx.sum(y_onehot * log_probs, axis=-1)
