import mlx
import mlx.core as mx

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Applies LeakyReLU activation to the input array.

        Args:
            x (mx.array): Input array of any shape.

        Returns:
            mx.array: Output array with LeakyReLU applied, same shape as input.
        """
        return mx.nn.leaky_relu(x, negative_slope=self.negative_slope)

batch_size = 16
dim = 16384

def get_inputs():
    x = mx.randn(batch_size, dim)
    return [x]

