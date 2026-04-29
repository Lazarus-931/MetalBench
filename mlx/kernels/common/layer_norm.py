# Layer Norm
import mlx
import mlx.core as mx


class Model(nn.Module):
    """
    Layer normalization:

        y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta

    where gamma and beta are learned per-feature parameters.
    """
    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ):
        super(Model, self).__init__()
        self.l_n = mx.nn.layernorm(dims)
        
        
    def forward(x: mx.array):
        """
        Args:
            x: Input array of shape (..., dims)

        Returns:
            Layer-normalized output
        """
        return self.l_n(x)
    

N = 1024

