import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Cross-entropy loss: per-row negative log-likelihood.
    Returns shape (N,); take mean() downstream for the scalar loss.
    Targets passed as one-hot float32 (N, C)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: mx.array, y_onehot: mx.array) -> mx.array:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.sum(y_onehot * log_probs, axis=-1)
