from .Equation  import Equation
from .Operators import Operators
import tensorflow as tf

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        
    def compute_residual(self, x, forward_pass):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)
        """
        x = tf.convert_to_tensor(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u, f = forward_pass(x, split = True)
            lap = Operators.laplacian_vector(tape, u, x, self.comp_dim.n_out_sol)
        return lap + f

    def pre_process(self, dataset):
        """
        Pre-process in Laplace problem is the identity transformation
        """
        return dataset

    def post_process(self, outputs):
        """
        Post-process in Laplace problem is the identity transformation
        """
        return outputs