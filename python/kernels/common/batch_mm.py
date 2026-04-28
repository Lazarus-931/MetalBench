import mlx
import mlx.core as mx


class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input array of shape (batch_size, m, k).
            B: Input array of shape (batch_size, k, n).

        Returns:
            C: Output array of shape (batch_size, m, n).
        """
        return mx.bmm(A, B)

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = mx.randn(batch_size, m, k)
    B = mx.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
