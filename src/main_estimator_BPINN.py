import tensorflow as tf
import numpy as np

from fork.my_HMC import my_HMC
from networks.Theta import Theta

# check if GPU is being used. On my machine, it isn't
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")


def full_deriv(model, x):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = model(x)
  J_x = tape.batch_jacobian(y,x)
  return J_x

def time_deriv(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  J_x = full_deriv(model, x)
  return J_x[:,:,0]  # we only want derivative with respect to time, wich is the first variable

def time_deriv_2(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  with tf.GradientTape(persistent=True) as tape1:
    tape1.watch(x)
    with tf.GradientTape(persistent=True) as tape2:
      tape2.watch(x)
      y2 = model(x)
    J_x_2 = tape2.batch_jacobian(y2,x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable

  H_x_2 = tape1.batch_jacobian(J_x_2, x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable
  return J_x_2, H_x_2

# to put in a "dynamics" file?
def dynamics_oscillator(x, k):
  F = np.asarray([[0.0, 1.0],[-k, 0.0]], dtype=np.float32).transpose()
  return x @ F

par_method = {'epochs': 100, 
              'burn_in': 0, 
              'HMC_skip': 10, 
              'HMC_L': 100,  # number of leapfrog steps per sample of theta
              'HMC_dt': 1e-4,  # time step length for leapfrog steps
              'HMC_eta': 0.5*1e-3  # mass of artificial Hamiltonian system
              }
  
# making HMC_eta smaller improved acceptance rate, but I would guess has consequences in terms of effectively sampling the distribution

par_nn = {'n_state': 2, # number of state variables
          'n_obs': 1, # number of variables per observation
          'n_obs_sequence': 3, # number of observations in a sequence fed to the model
          'n_params': 1, # number of parameters to the dynamics
          'n_neurons': [50, 50], # number of neurons in each layer
          'activation_function': 'swish',  # must be something tf.keras.layers.Dense can accept as activation
          'dynamics_function': dynamics_oscillator,  # 
          'stddev': 1}  # stddev of initialisation of NN, but also used for prior in BayesNN formulation

# ## define specific neural network architecture
class EstimatorNN:
  def __init__(self, par_nn):
    self.phys_params = 2.0  # placeholder

    self.n_state = par_nn['n_state']  # number of state variables
    self.n_obs = par_nn['n_obs']  # number of variables per observation
    self.n_obs_sequence = par_nn['n_obs_sequence']  # number of observations in a sequence fed to the model
    self.n_params = par_nn['n_params']  # number of parameters to the dynamics
    self.n_neurons = par_nn['n_neurons']  # number of neurons in each layer
    self.activation_function = par_nn['activation_function']
    self.dynamics_function = par_nn['dynamics_function']  #
    self.stddev = par_nn['stddev']  # stddev of initialisation of NN, but also used for prior in BayesNN formulation
    
    self.n_inputs = 1 + self.n_params + self.n_obs*self.n_obs_sequence  # the 1 is for the time input variable, always assumed 0th
    self.n_outputs = self.n_state

    self.build_NN(seed=0)

  def phys_loss(self, x, k):
    f_nn = time_deriv(self.model, x)
    f_known = self.dynamics_function(self.model(x), k)

    # return f_nn, f_known
    return tf.math.reduce_sum(tf.square(f_nn-f_known))

  def loss_total(self, dataset, full_loss = True):
    # TODO: must divide by the variance of measurements V
    # TODO: implement other losses
    V = 1.0  # placeholder
    phl = self.phys_loss(dataset, self.phys_params) / (2.0*V)
    phTot = phl
    return phTot
  
  # idential to the one present in LossNN
  def grad_loss(self, dataset, full_loss = True):
    # fl = lambda x : self.loss_total(x, full_loss)
    # J, _ = tf_gradient_wrapper(fl, dataset, self.model.trainable_variables)
    # return J
    """ Computation of the gradient of the loss function with respect to the network trainable parameters """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(self.model.trainable_variables)
        diff_llk = self.loss_total(dataset, full_loss)
    grad_thetas = tape.gradient(diff_llk, self.model.trainable_variables)
    return Theta(grad_thetas)
  
  # code based on CoreNN
  @property
  def nn_params(self):
    # """ Getter for nn_params property """
    weights = [layer.get_weights()[0] for layer in self.model.layers]
    biases  = [layer.get_weights()[1] for layer in self.model.layers]
    theta = list()
    for w, b in zip(weights, biases):
        theta.append(w)
        theta.append(b)
    return Theta(theta)
  
  # version below doesn't copy data for some reason but above seems to...
    # return Theta(self.model.trainable_variables)

  @nn_params.setter
  def nn_params(self, theta):
      """ Setter for nn_params property """
      for layer, weight, bias in zip(self.model.layers, theta.weights, theta.biases):
          layer.set_weights((weight,bias))

  # code modified from CoreNN
  def build_NN(self, seed):
    """
    Initializes a fully connected Neural Network with 
    - Random Normal initialization of weights
    - Zero initialization for biases
    """

    # Set random seed for inizialization
    tf.random.set_seed(seed)
    # Input Layer
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(self.n_inputs,)))
    # Hidden Layers
    for n_neurons in self.n_neurons:
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
        model.add(tf.keras.layers.Dense(n_neurons, activation=self.activation_function, 
                  kernel_initializer=initializer, bias_initializer='zeros'))
    # Output Layer
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
    model.add(tf.keras.layers.Dense(self.n_outputs, 
                  kernel_initializer=initializer, bias_initializer='zeros'))

    self.model = model



estimator_model = EstimatorNN(par_nn)

n_samples = 100
xr = tf.random.normal(shape=(n_samples,estimator_model.n_inputs))

y = estimator_model.loss_total(xr)
J = estimator_model.grad_loss(xr)

hmc = my_HMC(estimator_model, par_method, data_batch=xr, debug_flag=True)

import time

theta_list = []
start = time.time()
for _ in range(10):
  theta1 = hmc.sample_theta(estimator_model.nn_params)
  theta_list.append(theta1)
end = time.time()
print("time elapsed: ", end - start)

# th0 = estimator_model.nn_params
# l0 = estimator_model.loss_total(xr)
# estimator_model.nn_params -= estimator_model.grad_loss(xr)*1e-4
# th1 = estimator_model.nn_params
# lf = estimator_model.loss_total(xr)
# print(lf-l0)
