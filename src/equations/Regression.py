from .Equation  import Equation

class Regression(Equation):
    """
    Regression implementation
    """
    def __init__(self, par):
        super().__init__(par)

    def comp_residual(self, *_):
        raise Exception("There's no PDE in Regression Problem!")
