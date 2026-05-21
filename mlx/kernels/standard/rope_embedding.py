import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """RoPE: rotary position embedding. Input x is (S, D); D must be even.
    For each position p and dim-pair (2i, 2i+1):
        out[p, 2i]   = x[p, 2i]   * cos(p*ω_i) − x[p, 2i+1] * sin(p*ω_i)
        out[p, 2i+1] = x[p, 2i]   * sin(p*ω_i) + x[p, 2i+1] * cos(p*ω_i)
    where ω_i = 1 / 10000^(2i / D).
    """
    def __init__(self, base: float = 10000.0):
        super(Model, self).__init__()
        self.base = base

    def forward(self, x: mx.array) -> mx.array:
        S, D = x.shape
        half = D // 2
        idx = mx.arange(half, dtype=mx.float32)
        omega = 1.0 / (self.base ** (2.0 * idx / D))
        pos = mx.arange(S, dtype=mx.float32)
        angles = pos[:, None] * omega[None, :]
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        x_pair = x.reshape(S, half, 2)
        x0 = x_pair[:, :, 0]
        x1 = x_pair[:, :, 1]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return mx.stack([y0, y1], axis=-1).reshape(S, D)
