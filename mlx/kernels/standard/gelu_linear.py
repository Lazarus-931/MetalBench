import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """GELU + Linear: y = gelu(x @ W). BERT/GPT-2 FFN block."""
    def __init__(self, dim=1024, out_dim=1024):
        super(Model, self).__init__()
        self.linear = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        return nn.gelu(x @ w)
