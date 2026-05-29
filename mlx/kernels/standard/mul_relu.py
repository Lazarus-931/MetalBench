import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """y = relu(x * a)

    Inputs: x (N,), a (N,). Output: (N,).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, a):
        return nn.relu(x * a)
