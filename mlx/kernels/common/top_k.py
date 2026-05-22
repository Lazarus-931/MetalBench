import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    def __init__(self, k: int = 16):
        super(Model, self).__init__()
        self.k = k

    def forward(self, x: mx.array) -> mx.array:
        topk_vals = mx.topk(x, self.k, axis=-1)
        return mx.sort(topk_vals, axis=-1)[..., ::-1]
