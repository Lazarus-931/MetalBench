import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """KL divergence term element-wise: target*(log target - log_pred). Inputs are
    abs'd so random-normal test data stays positive."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, log_pred: mx.array, target: mx.array) -> mx.array:
        t = mx.abs(target) + 1e-6
        lp = -mx.abs(log_pred)
        return t * (mx.log(t) - lp)
