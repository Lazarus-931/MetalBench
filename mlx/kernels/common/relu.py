# ReLU

import mlx
import mlx.core as mx

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Applies ReLU activation to the input array.

        Args:
            x (mx.array): Input array of any shape.

        Returns:
            mx.array: Output array with ReLU applied, same shape as input.
        """
        return mx.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = mx.randn(batch_size, dim)
    return [x]

