import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Full LLaMA decoder layer: RMSNorm + GQA(+RoPE) + residual + RMSNorm
    + SwiGLU FFN + residual.

    Inputs:
        x       (S, D)
        W_qkv   (D, D + 2*H_kv*D_head)
        W_o     (D, D)
        W_gu    (D, 2*FF)        — fused gate || up for SwiGLU
        W_down  (FF, D)
    Output:  (S, D)
    """
    def __init__(self, S=64, D=128, H=4, H_kv=2, FF=256, base=10000.0, eps=1e-5):
        super(Model, self).__init__()
        self.S, self.D, self.H, self.H_kv, self.FF = S, D, H, H_kv, FF
        self.base, self.eps = base, eps
        self.D_head = D // H
        self.G = H // H_kv

    def forward(self, x, W_qkv, W_o, W_gu, W_down):
        S, D, H, H_kv, Dh, FF = self.S, self.D, self.H, self.H_kv, self.D_head, self.FF

        # ---- attention ----
        h = self._rms_norm(x)
        qkv = h @ W_qkv
        q = qkv[:, :D].reshape(S, H, Dh).transpose(1, 0, 2)
        k = qkv[:, D:D + H_kv*Dh].reshape(S, H_kv, Dh).transpose(1, 0, 2)
        v = qkv[:, D + H_kv*Dh:].reshape(S, H_kv, Dh).transpose(1, 0, 2)
        q = self._rope(q); k = self._rope(k)
        k = mx.repeat(k, self.G, axis=0); v = mx.repeat(v, self.G, axis=0)
        scores = (q @ k.transpose(0, 2, 1)) / mx.sqrt(mx.array(float(Dh)))
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(1, 0, 2).reshape(S, D) @ W_o
        x = x + out

        # ---- SwiGLU FFN ----
        h = self._rms_norm(x)
        gu = h @ W_gu                                         # (S, 2*FF)
        gate = nn.silu(gu[:, :FF])
        up = gu[:, FF:]
        ff = (gate * up) @ W_down                             # (S, D)
        return x + ff

    def _rms_norm(self, x):
        v = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(v + self.eps)

    def _rope(self, t):
        heads, S, Dh = t.shape
        half = Dh // 2
        idx = mx.arange(half, dtype=mx.float32)
        omega = 1.0 / (self.base ** (2.0 * idx / Dh))
        pos = mx.arange(S, dtype=mx.float32)
        ang = pos[:, None] * omega[None, :]
        c, s = mx.cos(ang), mx.sin(ang)
        t_pair = t.reshape(heads, S, half, 2)
        t0, t1 = t_pair[..., 0], t_pair[..., 1]
        r0 = t0 * c - t1 * s
        r1 = t0 * s + t1 * c
        return mx.stack([r0, r1], axis=-1).reshape(heads, S, Dh)
