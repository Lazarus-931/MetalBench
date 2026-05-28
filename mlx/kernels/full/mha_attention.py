import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Multi-head attention forward: fused QKV + scaled dot-product + softmax + output proj.

    Single batch. Inputs: x (S,D), wq,wk,wv,wo each (D,D). H=4, Dh=D//H.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, wq, wk, wv, wo):
        S, D = x.shape
        H = 4
        Dh = D // H
        scale = 1.0 / mx.sqrt(mx.array(Dh, dtype=mx.float32))

        q = (x @ wq).reshape(S, H, Dh).transpose(1, 0, 2)   # (H,S,Dh)
        k = (x @ wk).reshape(S, H, Dh).transpose(1, 0, 2)
        v = (x @ wv).reshape(S, H, Dh).transpose(1, 0, 2)

        scores = (q @ k.transpose(0, 2, 1)) * scale          # (H,S,S)
        attn = mx.softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(1, 0, 2).reshape(S, D)    # (S,D)
        return ctx @ wo
