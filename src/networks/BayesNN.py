from .PredNN import PredNN
from .LossNN import LossNN
from .equations import Laplace

class BayesNN(PredNN, LossNN):

    def __init__(self, par):
        
        self.trained = False
        self.seed    = par.utils["random_seed"] 
        self.losses  = self.__initialize_losses()

        equation  = self.__initialize_equation(par)
        comp_res  = equation.compute_residual
        pre_proc  = equation.pre_process
        post_proc = equation.post_process
        
        super(BayesNN,self).__init__(par=par, comp_res=comp_res,
                                     pre=pre_proc, post=post_proc)

    def __initialize_losses(self):
        
        keys = ("Total", "res", "data", "prior")
        loss_dict, logloss_dict = dict(), dict()

        for key in keys: 
            loss_dict[key]    = list()
            logloss_dict[key] = list()

        return (loss_dict, logloss_dict)

    def __initialize_equation(self, par):

        equation = par.experiment["dataset"]
        if   equation == "laplace1D_cos": return Laplace(par)
        elif equation == "laplace2D_cos": return Laplace(par)
        else: raise("Equation not implemeted!")

    def loss_step(self, new_losses):
        
        keys = ("Total", "res", "data", "prior")
        for key in keys: 
            self.losses[0][key].append(new_losses[0][key])
            self.losses[1][key].append(new_losses[1][key])
