import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """SiLU + Linear: y = silu(x @ W). Fused gating projection."""
    def __init__(self, dim=1024, out_dim=1024):
        super(Model, self).__init__()
        self.linear = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        return nn.silu(x @ w)
