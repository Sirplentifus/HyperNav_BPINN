from .CoreNN import CoreNN
import numpy as np
import tensorflow as tf

class PredNN(CoreNN):
    
    def __init__(self, pre, post, **kw):
        
        super(PredNN, self).__init__(**kw)

        # Functions for pre and post processing of inputs and outputs of the network
        self.pre_process = pre
        self.post_process = post
        
        # Empty list where samples of network parameters will be stored
        self.thetas = list() 

    def __compute_sample(self, theta, inputs):
        """ 
        Computes output sampling with one given theta
        """
        self.nn_params = theta
        sample = self.forward(inputs, split=True)

        return self.post_process(sample)
    
    def __predict(self, inputs, n_thetas = None):
        """ 
        Computes a list of output sampling from the list of thetas
        out_sol: list with len n_thetas, of tf tensor (n_sample, n_out_sol)
        out_par: list with len n_thetas, of tf tensor (n_sample, n_out_par)
        """
        inputs = self.pre_process(inputs)
        out_sol = list()
        out_par = list()

        if n_thetas is None: n_thetas = len(self.thetas)
        if n_thetas > len(self.thetas): raise("Not enough thetas!!!")
        for theta in self.thetas[-n_thetas:]:
            outputs = self.__compute_sample(theta, inputs)
            out_sol.append(outputs[0])
            out_par.append(outputs[1])

        return out_sol, out_par

    def __statistics(self, output):
        """
        Returns mean and standard deviation of a list of tensors
        mean : (n_samples, output[0].shape[1])
        std  : (n_samples, output[0].shape[1])
        """
        mean = np.mean(output, axis = 0)
        std = np.std(output, axis = 0)
        return mean, std

    def mean_and_std(self, inputs):
        """
        Computes mean and standard deviation of the output samples
        functions_confidence: dictionary containing outputs of __statistics() for out_sol and out_par
        """
        out_sol, out_par = self.__predict(inputs)
        mean_sol, std_sol = self.__statistics(out_sol)
        mean_par, std_par = self.__statistics(out_par)
        functions_confidence = {"sol_NN": mean_sol, "sol_std": std_sol, 
                                "par_NN": mean_par, "par_std": std_par}
        return functions_confidence

    def draw_samples(self, inputs):
        out_sol, out_par = self.__predict(inputs)
        out_sol = [value.numpy() for value in out_sol]
        out_par = [value.numpy() for value in out_par]
        functions_nn_samples = {"sol_samples": out_sol, "par_samples": out_par}
        return functions_nn_samples

    def __compute_UQ(self, function_confidence):
        
        u_q = dict()

        u_q["uq_sol_mean"] = np.mean(function_confidence["sol_std"], axis = 0)
        u_q["uq_par_mean"] = np.mean(function_confidence["par_std"], axis = 0)
        u_q["uq_sol_max"]  = np.max(function_confidence["sol_std"], axis = 0)
        u_q["uq_par_max"]  = np.max(function_confidence["par_std"], axis = 0)
        
        return u_q

    def __metric(self, x, y):
        metric = tf.keras.losses.MSE 
        return [metric(x[:,i],y[:,i]).numpy() for i in range(x.shape[1])]

    def __compute_errors(self, function_confidence, dataset):

        sol_true, par_true = dataset.dom_data[1], dataset.dom_data[2]
        sol_NN, par_NN = function_confidence["sol_NN"], function_confidence["par_NN"]

        error_sol = self.__metric(sol_NN, sol_true)
        error_par = self.__metric(par_NN, par_true)
        norm_sol  = self.__metric(sol_true, tf.zeros_like(sol_true))
        norm_par  = self.__metric(par_true, tf.zeros_like(par_true))

        err = dict()
        err["error_sol"] = np.divide(error_sol,norm_sol)
        err["error_par"] = np.divide(error_par,norm_par)

        return err
    
    def test_errors(self, function_confidence, dataset):
        err = self.__compute_errors(function_confidence, dataset)
        u_q = self.__compute_UQ(function_confidence)
        return err | u_q

    def disp_errors(self, message, values):
        """ Format of the error printing """
        if len(values) == 1:
            print(f"   {message}: {values[0]:1.3e}")
        else:
            labels = ["x", "y", "z"][:len(values)]
            for label, value in zip(labels, values):
                print(f"   {message} in direction {label}: {value:1.3e}")

    def show_errors(self, errors):
        """ Print on terminal the errors computed above """
        self.disp_errors("Relative sol error", errors["error_sol"])
        self.disp_errors("Relative par error", errors["error_par"])
        self.disp_errors("Mean UQ sol", errors["uq_sol_mean"])
        self.disp_errors("Mean UQ par", errors["uq_par_mean"])
        self.disp_errors("Max UQ sol", errors["uq_sol_max"])
        self.disp_errors("Max UQ par", errors["uq_par_max"])