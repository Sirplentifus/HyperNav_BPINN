from argparse import ArgumentParser

class Parser(ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian PINN for PDEs')

        # Configuration (choose configuration file to set other parameters)
        self.add_argument('--config', type=str, default="default", 
                        help="""Name of json file where we can find all the parameters. 
    							You have to provide the parameters for at least the method you've selected.
        						You can also overwrite some parameters directly from terminal
                            """)
        # Problem (choose the physical problem)
        self.add_argument('--problem', type=str, 
                        help="""Choose the experiment :
                                - laplace1D (1D, Laplace)
                                - laplace2D (2D, Laplace)
                            """)
        # Dataset (choose the data for the problem)
        self.add_argument('--case_name', type=str, 
                        help="""Choose the experiment :
                                - cos (1D-2D, Laplace)
                            """)
        # Algorithm (choose training algorithm)
        self.add_argument('--method', type=str, 
                        help="""Methods to use for BPINN. Available:
        		                - HMC  (Hamiltonian Monte Carlo)
        			            - SVGD (Stein Variational Gradient Descent)
                                - VI   (variational inference)
                                - TEST (use for debug purpouses)
                            """)

        # Experiment ??? SISTEMARE COMMENTI
        self.add_argument('--num_sol', type=int,   help="Needs to be integer. Number of Domain Data to use as fitting data for solution")
        self.add_argument('--num_par', type=int,   help="Needs to be integer. Number of Domain Data to use as fitting data for parametric field")
        self.add_argument('--num_bnd', type=int,   help="Needs to be integer. Number of Domain Data to use as boundary data")
        self.add_argument('--num_pde', type=int,   help="Needs to be integer. Number of Domain Data to use as collocation points")


        self.add_argument('--var_sol', type=float, help="Artificial noise in exact dataset for solution")
        self.add_argument('--var_par', type=float, help="Artificial noise in exact dataset for parametric field")
        self.add_argument('--var_bnd', type=float, help="Artificial noise in exact dataset for boundary data")
        self.add_argument('--var_pde' ,type=float, help="Artific")
        
        # Architecture
        self.add_argument('--activation', type=str, help='Activation function for hidden layers')
        self.add_argument('--n_layers'  , type=int, help='Number of hidden layers in the NN')
        self.add_argument('--n_neurons' , type=int, help='Number of neurons in each hidden layer in the NN')

        # Utils
        self.add_argument('--random_seed', type=int,  help='Random seed for np and tf random generator')
        self.add_argument('--debug_flag' , type=bool, help='Prints general debug utilities at each iteration')
        self.add_argument('--save_flag'  , type=bool, help='Flag to save results in a new folder')
        self.add_argument('--gen_flag'   , type=bool, help='Flag for new data generation')

        # %% Algoritm Parameters
        self.add_argument('--epochs', type=int, help='Number of epochs to train')

        # HMC
        self.add_argument('--burn-in', type=int,   help="Number of samples to use in HMC (after burn-in). Needs to be <= epochs")
        self.add_argument('--HMC_L'  , type=int,   help="L: number of leapfrog step in HMC")
        self.add_argument('--HMC_dt' , type=float, help="dt: step size in HMC")

        # SVGD

        # VI


