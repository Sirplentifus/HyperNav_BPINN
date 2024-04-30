import tensorflow as tf
import numpy as np

# following lines suppress warnings, which we perhaps should eventually look into
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.ERROR)

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
var.assign_add([1, 1, 1])

x = tf.Variable(1.0)
def f(x):
  y = x**2 + 2*x - 5
  return y

f(x)

with tf.GradientTape(persistent=True) as tape:
  y = f(x)

g_x = tape.gradient(y, x)  # g(x) = dy/dx

g_x

## start of serious business

seed = 0

n_state = 2  # number of state variables
n_obs  = 1  # number of variables per observation
n_obs_sequence = 3  # number of observations in a sequence fed to the model

n_inputs = 1 + n_state + n_obs*n_obs_sequence  # the 1 is for time

n_outputs = n_state

n_layers = 2
n_neurons = 50

activation_function = 'swish'  # Cannot be a ReLU because the NN must be differentiable! Swish is like a smooth ReLU it seems. It's the same thing as a SiLU but with a parameter

stddev = 1

## based on code copy pasted from CoreNN.__build_NN

# Set random seed for inizialization
tf.random.set_seed(seed)
# Input Layer
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(n_inputs,)))
# Hidden Layers
for _ in range(n_layers):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
    model.add(tf.keras.layers.Dense(n_neurons, activation=activation_function, 
                kernel_initializer=initializer, bias_initializer='ones'))
# Output Layer
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
model.add(tf.keras.layers.Dense(n_outputs, 
                kernel_initializer=initializer, bias_initializer='ones'))

print(f"{isinstance(model, tf.Module)=}")

## try to get gradient of output with respect to time (0-th input)

x = tf.Variable(np.zeros((1, n_inputs)))
# x = tf.zeros((1, n_inputs))

y0 = model(x)

with tf.GradientTape(persistent=True) as tape:
    y = model(x)
    y0 = y[0,0]

# exit()

# this way we get the gradient of the sum
# g_x = tape.gradient(y, x)  

# to get a Jacobian you do
J_x = tape.jacobian(y,x)

# F.D. for comparison:
x1 = tf.Variable([[1e-8, 0, 0, 0, 0, 0]])
(model(x1)-model(x))/1e-8 

# to get the model parameters
weights = [layer.get_weights()[0] for layer in model.layers]
biases  = [layer.get_weights()[1] for layer in model.layers]

# this is what model(x) does:
x2 = tf.Variable([[1, 3.0, 1.1, 0, 0, 0]])
h_list = []
h_list.append(x2.numpy())
for i in range(len(model.layers)):
  h_list.append(tf.nn.swish(h_list[-1]@weights[i] + biases[i]))
print(h_list[-1])
print(model(x2))
print("------------")

# look at cases with multiple samples
# https://www.tensorflow.org/guide/advanced_autodiff#batch_jacobian 

def full_deriv(model, x):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = model(x)
  J_x = tape.batch_jacobian(y,x)
  return J_x

def time_deriv(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  J_x = full_deriv(model, x)
  return J_x[:,:,0]  # we only want derivative with respect to time, wich is the first variable

# higher order deriv

with tf.GradientTape(persistent=True) as tape1:
  tape1.watch(x)
  with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(x)
    y2 = model(x)
  J_x_2 = tape2.batch_jacobian(y2,x)[:,:,0]

H_x_2 = tape1.batch_jacobian(J_x_2, x)[:,:,0]

def time_deriv_2(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  with tf.GradientTape(persistent=True) as tape1:
    tape1.watch(x)
    with tf.GradientTape(persistent=True) as tape2:
      tape2.watch(x)
      y2 = model(x)
    J_x_2 = tape2.batch_jacobian(y2,x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable

  H_x_2 = tape1.batch_jacobian(J_x_2, x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable
  return J_x_2, H_x_2

J_x_2_ = time_deriv(model, x2)
J_x_2, H_x_2 = time_deriv_2(model, x2)

# version where model also explicitely predicts velocity
# def ode_oscillator(model, x, k):
#   J_x_2 = time_deriv(model, x)
#   # r = x[:,0]
#   # v = x[:,1]
#   # rdot = J_x_2[:,0]
#   # vdot = J_x_2[:,1]
#   # rdot-v
#   # vdot - (-k*r)

#   F = np.asarray([[0.0, 1.0],[-k, 0.0]], dtype=np.float32).transpose()
#   f = J_x_2 @ F
#   return f

def phys_loss(model, x, k):
  f_nn = time_deriv(model, x)

  F = np.asarray([[0.0, 1.0],[-k, 0.0]], dtype=np.float32).transpose()
  f_known = model(x) @ F

  # return f_nn, f_known
  return tf.math.reduce_mean(tf.square(f_nn-f_known))

xr = tf.random.normal(shape=(5,6))
#tape.watch not needed if variable is a tf.Variable, but whatever
# xr = tf.Variable(xr)

lph = phys_loss(model, xr, 2.0)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(model.trainable_variables)  # I think this is unnecessary because model.trainable_variables are tf.Variable 
  lph = phys_loss(model, xr, 2.0)
J_lph_theta = tape.gradient(lph, model.trainable_variables)

# def flatten_tensor_list(tensor_list):
#   flattened_tensors = list(map(lambda t : tf.reshape(t, (-1)), tensor_list))
#   return tf.concat(flattened_tensors, axis=0)

from networks.Theta import Theta

# this wrapper, plus the functions above, kind of negates the need for the complicated class structure for NNs I think...
def tf_gradient_wrapper(function, x, x2=x): # function is evaluated on x, but we get the gradient with respect to x2, which may be a "hidden" parameter of function
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x2)
    y = function(x)
  J = Theta(tape.gradient(y, x2))
  return J, y

J_lph_theta_, *_ = tf_gradient_wrapper(lambda x: phys_loss(model, x, 2.0), xr, model.trainable_variables)

# flatten_tensor_list(J_lph_theta_)

def assign_params(model, theta):
  for i in range(len(model.trainable_variables)):
        model.trainable_variables[i].assign(theta[i])
        # model.trainable_variables[i] = theta[i]

def leapfrog_step(model, data_batch, loss_function, r, dt):
    """ Performs one leap-frog step starting from previous values of theta/sigma and r/s """
    
    grad_theta, *_ = tf_gradient_wrapper(loss_function, data_batch, model.trainable_variables)
    r = r - grad_theta * dt / 2
    new_theta = Theta(model.trainable_variables) + r * dt  # could avoid some allocations here
    assign_params(model, new_theta.values)
    grad_theta, *_ = tf_gradient_wrapper(loss_function, data_batch, model.trainable_variables)
    r = r - grad_theta * dt / 2

    return new_theta, r

    # grad_theta = model.grad_loss(data_batch, loss_function)
    # r = r - grad_theta * dt / 2
    # model.nn_params = old_theta + r * dt 
    # grad_theta = model.grad_loss(data_batch, loss_function)
    # r = r - grad_theta * dt / 2
    # return model.nn_params, r

theta = Theta(model.trainable_variables)

print(theta.values[0][0,0])

r0 = theta.normal()
dt = 0.1
theta, r = leapfrog_step(model, xr, lambda x: phys_loss(model, x, 2.0), r0, dt)

print(theta.values[0][0,0])
print(model.trainable_variables[0][0,0])

# def compute_alpha(h0,  h1):
#     """ Computation of acceptance probabilities alpha and sampling of p (logarithm of both quantities) """
#     p     = np.log(np.random.uniform())
#     alpha = min(0, -h1+h0) 
#     if np.isnan(h1): alpha = float("-inf")
#     return alpha, p

# def hamiltonian(self, theta, r):
#     """ Evaluation of the Hamiltonian function """
#     self.model.nn_params = theta
#     u = self.model.loss_total(self.data_batch, self.__full_loss).numpy()
#     v_r = r.ssum() * self.HMC_eta**2/2
#     return u + v_r

from fork.my_HMC import my_HMC

# to put in a "dynamics" file?
def dynamics_oscillator(x, k):
  F = np.asarray([[0.0, 1.0],[-k, 0.0]], dtype=np.float32).transpose()
  return x @ F

debug_flag = False
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
          'dynamics_function': 'dynamics_oscillator',  # name of a dynamics function, which must be accessible in this file (so make sure it's imported)
          'stddev': 1}  # stddev of initialisation of NN, but also used for prior in BayesNN formulation

# auxiliary function
def normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        return 0.5 * n * ( mse * tf.math.exp(log_var) - log_var)

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
    self.dynamics_function = globals()[par_nn['dynamics_function']],  # name of a dynamics function, which must be accessible in this file (so make sure it's imported) - globals() produces a dictionary with the symbols defined in the global environment, which should include the function named in this parameter
    self.stddev = par_nn['stddev']  # stddev of initialisation of NN, but also used for prior in BayesNN formulation
    
    self.n_inputs = 1 + self.n_params + self.n_obs*self.n_obs_sequence  # the 1 is for the time input variable, always assumed 0th
    self.n_outputs = self.n_state

    self.build_NN(seed=0)

  def phys_loss(self, x, k):
    f_nn = time_deriv(self.model, x)
    f_known = self.dynamics_function(x)

    # return f_nn, f_known
    return tf.math.reduce_sum(tf.square(f_nn-f_known))

  def loss_total(self, dataset, full_loss = True):
    # TODO: must divide by the variance of measurements V
    # TODO: implement other losses
    V = 1.0  # placeholder
    phl = phys_loss(self.model, dataset, self.phys_params) / (2.0*V)
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
