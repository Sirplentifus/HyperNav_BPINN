import numpy as np
import tensorflow as tf

class CoreNN():

    """
    ***** Key Features *****
    - Build initial network
    - Contain layers and weights and biases (theta)
    - Do forward pass on given theta

    **** Other Features **** WIP
    - Save and Load single theta?
    - Architecture Recap ?
    
    Neural networks parameters (theta)
        - nn_params = (weights, biases)
        - weights is a list of numpy arrays 
            - 1 layer               : shape (n_input, n_neurons)
            - (n_layers - 1) layers : shape (n_neurons, n_neurons) 
            - 1 layer               : shape (n_neurons, n_out_sol+n_out_par)
        - biases is a list of numpy arrays 
            - n_layers layers       : shape (n_neurons,) 
            - 1 layer               : shape (n_out_sol+n_out_par,)
    """
    
    def __init__(self, par):

        # Domain dimensions
        self.n_inputs  = par.comp_dim.n_input
        self.n_out_sol = par.comp_dim.n_out_sol
        self.n_out_par = par.comp_dim.n_out_par

        # Architecture parameters
        self.n_layers   = par.architecture["n_layers"]
        self.n_neurons  = par.architecture["n_neurons"]
        self.activation = par.architecture["activation"]

        # Save parameters for child classes
        self.par = par
        # Build the Neural network architecture
        self.model = self.__build_NN()

    def __build_NN(self):
        """
        Initializes a fully connected Neural Network with 
        - Glorot Uniform initialization of weights
        - Zero initialization for biases
        """

        # Input Layer
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.n_inputs,)))
        # Hidden Layers
        for _ in range(self.n_layers):
            model.add(tf.keras.layers.Dense(self.n_neurons, activation=self.activation, 
                      kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        # Output Layer
        model.add(tf.keras.layers.Dense(self.n_out_sol+self.n_out_par, 
                  kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        return model
        
    @property
    def nn_params(self):
        """ Getter for nn_params property """
        weights = [layer.get_weights()[0] for layer in self.model.layers]
        biases  = [layer.get_weights()[1] for layer in self.model.layers]
        return (weights, biases)

    @nn_params.setter
    def nn_params(self, theta):
        """ Setter for nn_params property """
        weights, biases = theta
        for layer, weight, bias in zip(self.model.layers, weights, biases):
            layer.set_weights((weight,bias))

    def forward(self, inputs, split = False):
        """ 
        Simple prediction on Solution and Parametric field 
        inputs : np array  (n_samples, n_input)
        [split = False] output : tf tensor (n_samples, n_out_sol + n_out_par)
        [split = True] out_sol : tf tensor (n_samples, n_out_sol)
        [split = True] out_par : tf tensor (n_samples, n_out_par)
        """
        x = tf.convert_to_tensor(inputs)
        # compute the output of NN at the inputs data
        output = self.model(x)
        if not split: return output

        # select solution output
        out_sol = output[:,:self.n_out_sol]
        # select parametric field output
        out_par = output[:,self.n_out_sol:]
        
        return out_sol, out_par
