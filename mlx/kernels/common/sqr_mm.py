# example1.py

import mlx
import numpy
import mlx.core as mx
from mlx import nn



class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: mx.array, B: mx.array) -> mx.array:
        """
        Performs the matrix multiplication.

        Args:
            A (mx.array): Input matrix A of shape (N, N).
            B (mx.array): Input matrix B of shape (N, N).

        Returns:
            mx.array: Output matrix C of shape (N, N).
        """
        return mx.matmul(A, B)

N = 2048

def get_inputs():
    A = mx.randn(N, N)
    B = mx.randn(N, N)
    return [A, B]

