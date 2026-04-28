
import mlx
import mlx.core as mx

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """
        Performs matrix multiplication.

        Args:
            A: Input array of shape (M, K).
            B: Input array of shape (K, N).

        Returns:
            Output array of shape (M, N).
        """
        return mx.matmul(A, B)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = mx.randn(M, K)
    B = mx.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

