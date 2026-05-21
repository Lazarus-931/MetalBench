import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Per-row argmax along the last dimension. Returns indices cast to f32
    so the harness (which assumes f32 outputs) can diff them."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: mx.array) -> mx.array:
        return mx.argmax(x, axis=-1).astype(mx.float32)
