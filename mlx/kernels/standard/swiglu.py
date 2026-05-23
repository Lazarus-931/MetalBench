import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """SwiGLU: silu(x @ W_gate) * (x @ W_up). LLaMA-style FFN gating."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array, w_gate: mx.array, w_up: mx.array) -> mx.array:
        return nn.silu(x @ w_gate) * (x @ w_up)
