import sys
import tensorflow as tf

## define Equation PINN is to learn
# from equations import Oscillator
from networks import BayesNN

## define specific neural network architecture
class EstimatorNN(BayesNN):
    def __init__(self, par, equation):
        super(EstimatorNN, self).__init__(par, equation)
    
    def __loss_residual(self, data):  # where the NN is derived with respect to inputs
        """ Physical loss; computation of the residual of the PDE """
        inputs = self.tf_convert(data["dom"])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            u, f = self.forward(inputs)
            residuals = self.pinn.comp_residual(inputs, u, f, tape)
        mse = self.__mse(residuals)
        log_var =  tf.math.log(1/self.vars["pde"]**2)
        log_res = self.__normal_loglikelihood(mse, inputs.shape[0], log_var)
        return mse, log_res
    
# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_dataset, switch_equation, switch_configuration

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import DataGenerator, Dataset    # Dataset Creation
# from networks import BayesNN                # Models
from algorithms import Trainer              # Algorithms

# %% Creating Parameters

starred_print("START")
configuration_file = "/best_models/HMC_osc_sin"
args   = Parser().parse_args()   # Load a param object from command-line
config = load_json(configuration_file)  # Load params from config file
params = Param(config, args)     # Combines args and config

# data_config = switch_dataset(params.problem, params.case_name)
# params.data_config = data_config

print(f"Bayesian PINN using {params.method}")
print(f"Solve the {params.inverse} problem of {params.pde} {params.phys_dim.n_input}D ")
starred_print("DONE")

# %% Model Building

print("Building the Model")
print(f"\tChosing {params.pde} equation...")
equation = switch_equation(params.problem)
print("\tInitializing the Bayesian PINN...")
bayes_nn = EstimatorNN(params, equation) # Initialize the Bayesian NN
starred_print("DONE")

