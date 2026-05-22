import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """LLaMA-style grouped-query attention with RoPE (no KV cache).

    Inputs:
        x       (S, D)                       — input
        W_qkv   (D, D + 2*H_kv*D_head)       — fused Q + grouped K/V projection
        W_o     (D, D)                       — output projection
    Output:  (S, D)

    Computes: RoPE(Q), RoPE(K); grouped attention with H query heads sharing
    H_kv key/value heads; softmax; @ V; project.
    """
    def __init__(self, S=64, D=128, H=4, H_kv=2, base=10000.0):
        super(Model, self).__init__()
        self.S, self.D, self.H, self.H_kv, self.base = S, D, H, H_kv, base
        self.D_head = D // H
        self.G = H // H_kv  # query heads per kv head

    def forward(self, x, W_qkv, W_o):
        S, D, H, H_kv, Dh = self.S, self.D, self.H, self.H_kv, self.D_head

        qkv = x @ W_qkv                                       # (S, D + 2*H_kv*Dh)
        q = qkv[:, :D].reshape(S, H, Dh).transpose(1, 0, 2)   # (H, S, Dh)
        k = qkv[:, D:D + H_kv*Dh].reshape(S, H_kv, Dh).transpose(1, 0, 2)
        v = qkv[:, D + H_kv*Dh:].reshape(S, H_kv, Dh).transpose(1, 0, 2)

        # RoPE on q and k (per-head, per-position)
        q = self._rope(q)
        k = self._rope(k)

        # Repeat K, V across query groups for matmul. (H, S, Dh)
        k = mx.repeat(k, self.G, axis=0)
        v = mx.repeat(v, self.G, axis=0)

        scores = (q @ k.transpose(0, 2, 1)) / mx.sqrt(mx.array(float(Dh)))
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        out = out.transpose(1, 0, 2).reshape(S, D)
        return out @ W_o

    def _rope(self, t):
        # t: (heads, S, Dh) → rotate each (s, 2i, 2i+1) pair
        heads, S, Dh = t.shape
        half = Dh // 2
        idx = mx.arange(half, dtype=mx.float32)
        omega = 1.0 / (self.base ** (2.0 * idx / Dh))
        pos = mx.arange(S, dtype=mx.float32)
        ang = pos[:, None] * omega[None, :]                   # (S, half)
        c, s = mx.cos(ang), mx.sin(ang)
        t_pair = t.reshape(heads, S, half, 2)
        t0, t1 = t_pair[..., 0], t_pair[..., 1]
        r0 = t0 * c - t1 * s
        r1 = t0 * s + t1 * c
        return mx.stack([r0, r1], axis=-1).reshape(heads, S, Dh)
