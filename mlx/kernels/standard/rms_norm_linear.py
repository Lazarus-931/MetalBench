import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """RMSNorm + Linear: y = rms_norm(x) @ W. LLaMA/Mistral core block."""
    def __init__(self, dim=1024, out_dim=1024, eps=1e-5):
        super(Model, self).__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.linear = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x: mx.array, w: mx.array) -> mx.array:
        n = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        return n @ w.T
        
