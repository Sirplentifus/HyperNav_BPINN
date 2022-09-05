import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

class Plotter():
    """ 
    Class for plotting utilities:
    Methods:
        - plot_losses: plots MSE and log-likelihood History
        - plot_nn_samples: plots all samples of solution and parametric field
        - plot_confidence: plots mean and std of solution and parametric field
        - show_plot: enables plot visualization
    """
    
    def __init__(self, path_plot):
        
        self.path_plot = path_plot

    def __order_inputs(self, inputs):
        """ Sorting the input points by label """
        idx = np.argsort(inputs)
        inputs = inputs[idx]
        return inputs, idx

    def __save_plot(self, path, title):
        """ Auxiliary function used in all plot functions for saving """
        path = os.path.join(path, title)
        plt.savefig(path, bbox_inches = 'tight')

    def __plot_confidence_1D(self, x, func, title, label = ("",""), fit = None):
        """ Plots mean and standard deviation of func (1D case); used in plot_confidence """
        x, idx = self.__order_inputs(x)
        func = [f[idx] for f in func]

        plt.figure()
        plt.plot(x, func[0], 'r-',  label='true')
        plt.plot(x, func[1], 'b--', label='mean')
        plt.plot(x, func[1] - func[2], 'g--', label='mean-std')
        plt.plot(x, func[1] + func[2], 'g--', label='mean+std')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title(title)

    def __plot_nn_samples_1D(self, x, func, label = ("",""), fit = None):
        """ Plots all the samples of func; used in plot_nn_samples """
        x, idx = self.__order_inputs(x)

        plt.figure()
        blurring = 2/len(func[1])
        for func_sample in func[1]:
            plt.plot(x, func_sample[idx,0], 'b-', markersize=0.01, alpha=blurring)

        func_ex = func[0][idx]
        plt.plot(x, func_ex, 'r-', label='true')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title('Samples from ' + label[1] + ' reconstructed distribution')

    def __plot_train(self, losses, name, title):
        """ Plots all the loss history; used in plot_losses """
        plt.figure()
        x = list(range(1,len(losses['Total'])+1))
        if name[:-4] == "LogLoss":
            plt.plot(x, losses['Total'], 'k--', lw=2.0, alpha=1.0, label = 'Total')
        for key, value in losses.items():
            if key == "Total": continue
            plt.plot(x, value, lw=1.0, alpha=0.7, label = key)

        plt.title(f"History of {title}")
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend(prop={'size': 9})
        self.__save_plot(self.path_plot, title)

    def plot_confidence(self, dataset, functions):
        """ Plots mean and standard deviation of solution and parametric field samples """
        inputs, u_true, f_true = dataset.dom_data
        u_points, u_values, _  = dataset.exact_data_noise

        u = (u_true, functions['sol_NN'], functions['sol_std'])
        u_fit = (u_points, u_values)
        f = (f_true, functions['par_NN'], functions['par_std'])

        self.__plot_confidence_1D(inputs[:,0], u, 'Confidence interval for u(x)', label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_confidence.png')
        self.__plot_confidence_1D(inputs[:,0], f, 'Confidence interval for f(x)', label = ('x','f'))
        self.__save_plot(self.path_plot, 'f_confidence.png')

    def plot_nn_samples(self, dataset, functions):
        """ Plots all the samples of solution and parametric field """
        inputs, u_true, f_true = dataset.dom_data
        u_points, u_values, _  = dataset.exact_data_noise

        u = (u_true, functions['sol_samples'])
        u_fit = (u_points, u_values)
        f = (f_true, functions['par_samples'])

        self.__plot_nn_samples_1D(inputs[:,0], u, label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_nn_samples.png')
        self.__plot_nn_samples_1D(inputs[:,0], f, label = ('x','f'), fit = None)
        self.__save_plot(self.path_plot, 'f_nn_samples.png')

    def plot_losses(self, losses):
        """ Generates the plots of MSE and log-likelihood """
        self.__plot_train(losses[0], "Loss.png"   , "Mean Squared Error")
        self.__plot_train(losses[1], "LogLoss.png", "Loss (Log-Likelihood)")

    def plot_sigmas(self, sigma_d, sigma_r):
        self.__plot_sigma(sigma_d)
        self.__plot_sigma(sigma_r)

    def __plot_sigma(self, sigma):
        pass

    def __wait_input(self, key):
        """ Start a loop that will run until the user enters key """
        key_input = ''
        while key_input != key:
            key_input = input("Input Q to quit: ").upper()

    def show_plot(self):
        """ Shows the plots """
        plt.show(block = False)
        self.__wait_input('Q')
