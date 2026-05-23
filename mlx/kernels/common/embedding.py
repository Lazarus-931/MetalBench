import mlx.core as mx
from mlx import nn


class Model(nn.Module):
    """Embedding lookup: out[s] = table[indices[s]]. The first op of every
    transformer (token id → embedding vector).

    Inputs in registry order:
        indices  (S,)         — int-valued but stored as f32 because the harness
                                only routes f32 inputs through MLX baselines.
        table    (V, D)       — the vocab × hidden embedding matrix.
    Output: (S, D)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, indices: mx.array, table: mx.array) -> mx.array:
        # Quantize f32 to int32 (values are in [0, V)), then gather rows.
        idx = mx.clip(indices, 0, table.shape[0] - 1).astype(mx.int32)
        return table[idx]
