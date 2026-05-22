import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Pre-LN transformer block (BERT/ViT style).

    y = x + FFN(LN(x + MHA(LN(x))))

    Inputs (registry-positional):
        x      (S, D)        — input activations
        W_qkv  (D, 3*D)      — fused Q/K/V projection
        W_o    (D, D)        — attention output projection
        W_ff1  (D, FF)       — FFN up projection
        W_ff2  (FF, D)       — FFN down projection
    """
    def __init__(self, S=64, D=128, H=4, FF=256, eps=1e-5):
        super(Model, self).__init__()
        self.S, self.D, self.H, self.FF, self.eps = S, D, H, FF, eps
        self.D_head = D // H

    def forward(self, x, W_qkv, W_o, W_ff1, W_ff2):
        S, D, H, FF = self.S, self.D, self.H, self.FF
        Dh = self.D_head

        # ---- attention sub-block ----
        ln1 = self._layer_norm(x)                            # (S, D)
        qkv = ln1 @ W_qkv                                    # (S, 3*D)
        q = qkv[:, 0:D].reshape(S, H, Dh).transpose(1, 0, 2) # (H, S, Dh)
        k = qkv[:, D:2*D].reshape(S, H, Dh).transpose(1, 0, 2)
        v = qkv[:, 2*D:3*D].reshape(S, H, Dh).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / mx.sqrt(mx.array(float(Dh)))   # (H, S, S)
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v                                        # (H, S, Dh)
        out = out.transpose(1, 0, 2).reshape(S, D)            # (S, D)
        out = out @ W_o                                       # (S, D)
        x = x + out                                           # residual

        # ---- FFN sub-block ----
        ln2 = self._layer_norm(x)
        ff = nn.gelu(ln2 @ W_ff1)                             # (S, FF)
        ff = ff @ W_ff2                                       # (S, D)
        return x + ff                                         # residual

    def _layer_norm(self, x):
        m = mx.mean(x, axis=-1, keepdims=True)
        v = mx.var(x, axis=-1, keepdims=True, ddof=0)
        return (x - m) * mx.rsqrt(v + self.eps)
