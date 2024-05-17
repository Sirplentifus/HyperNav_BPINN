import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from copy import copy

# arrays are always batch size first, i.e., if we have 100 samples of a 2D state vector, they will be in a 100 by 2 matrix.

# the only file I am using mostly as it was of the original code
from networks.Theta import Theta

# I modified slightly the original HMC file into this one
from fork.my_HMC import my_HMC

# import dynamics functions (my own definition for the harmonic oscillator)
from fork.generate_data_harmonic_oscillator import trajectory_propagator, dynamics

# check if GPU is being used. On my machine, it isn't
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

@tf.function
def full_deriv(model, x):
  with tf.GradientTape(watch_accessed_variables=False) as tape:  # persistent=True, watch_accessed_variables=False
    tape.watch(x)
    y = model(x)
  J_x = tape.batch_jacobian(y,x)
  return J_x

@tf.function
def time_deriv(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  J_x = full_deriv(model, x)
  return J_x[:,:,0]  # we only want derivative with respect to time, wich is the first variable

@tf.function
def time_deriv_2(model, x):  # input and output must have shape n_sample X n_inputs and n_sample X n_outputs, and the return will be of shape n_sample X n_outputs
  with tf.GradientTape(persistent=True) as tape1:
    tape1.watch(x)
    with tf.GradientTape(persistent=True) as tape2:
      tape2.watch(x)
      y2 = model(x)
    J_x_2 = tape2.batch_jacobian(y2,x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable

  H_x_2 = tape1.batch_jacobian(J_x_2, x)[:,:,0]  # we only want derivative with respect to time, wich is the first variable
  return J_x_2, H_x_2

par_method = {'epochs': 5000, # number of samples
              'burn_in': 0, 
              'HMC_skip': 1,  # UNUSED
              'HMC_L': 10,  # number of leapfrog steps per sample of theta
              'HMC_dt': 1e-4,  # time step length for leapfrog steps
              'HMC_eta': 1e-3  # mass of artificial Hamiltonian system
              }
  
# making HMC_eta smaller improved acceptance rate, but I would guess has consequences in terms of effectively sampling the distribution

par_nn = {'n_state': 2, # number of state variables
          'n_obs': 1, # number of variables per observation
          'n_obs_sequence': 3, # number of observations in a sequence fed to the model
          'n_params': 1, # number of parameters to the dynamics
          'n_neurons': [50, 50], # number of neurons in each layer
          'activation_function': 'swish',  # must be something tf.keras.layers.Dense can accept as activation
          'dynamics_function': lambda x,t,u : dynamics(x,t,1.0),  # 
          'stddev': 1,  # stddev of initialisation of NN
          'std_prior': 1.0,  # standard deviation of prior
          'std_phys': [1.0, 1.0],  # standard deviation of state information, in units of state
          'std_data': [1.0, 1.0],  # standard deviation of state derivative information
          }  


# functions designed to use @tf.function graph optimisation

@tf.function
def loss_prior(model, var_prior):
  params = model.trainable_variables
  L = tf.constant(0.0)
  for pm in params:
    L += tf.reduce_sum(tf.square(pm))
  return L/(2.0*var_prior)

@tf.function
def loss_data(model, x, y, var_data):
  ypred = model(x)
  x_mu_diff = tf.math.reduce_sum(tf.square(y-ypred), axis=0)
  return tf.math.reduce_sum(x_mu_diff/var_data)/2.0

@tf.function
def loss_phys(model, x, u, state, var_phys, dynamics_function):
  f_nn = time_deriv(model, x)

  y = model(x)
  t = x[:, 0]
  f_known = dynamics_function(state, t, u)
  x_mu_diff = tf.math.reduce_sum(tf.square(f_nn-f_known), axis=0)
  # return f_nn, f_known
  return tf.math.reduce_sum(x_mu_diff/(var_phys))/2.0

# nearly idential to the one present in LossNN
@tf.function
def loss_total(model, x, u, state, y, dynamics_function, var_prior, var_data, var_phys, full_loss=True):
   L = loss_prior(model, var_prior)
   L += loss_data(model, x, y, var_data)
   if full_loss:
    L += loss_phys(model, x, u, state, var_phys, dynamics_function)
   return L

# nearly idential to the one present in LossNN
@tf.function
def grad_loss_total(model, *args):
  # Computation of the gradient of the loss function with respect to the network trainable parameters
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(model.trainable_variables)
    diff_llk = loss_total(model, *args)
  grad_thetas = tape.gradient(diff_llk, model.trainable_variables)
  return grad_thetas

# ## define specific neural network architecture
class EstimatorNN:
  def __init__(self, par_nn):

    self.n_state = par_nn['n_state']  # number of state variables
    self.n_obs = par_nn['n_obs']  # number of variables per observation
    self.n_obs_sequence = par_nn['n_obs_sequence']  # number of observations in a sequence fed to the model
    self.n_params = par_nn['n_params']  # number of parameters to the dynamics
    self.n_neurons = par_nn['n_neurons']  # number of neurons in each layer
    self.activation_function = par_nn['activation_function']
    self.dynamics_function = par_nn['dynamics_function']  #
    self.stddev = par_nn['stddev']  # stddev of initialisation of NN
    
    self.n_inputs = 1 + self.n_params + self.n_obs*self.n_obs_sequence  # the 1 is for the time input variable, always assumed 0th
    self.n_outputs = self.n_state

    self.var_prior = tf.constant(par_nn['std_prior'])**2
    self.var_phys =  tf.constant(par_nn['std_phys'])**2
    self.var_data =  tf.constant(par_nn['std_data'])**2

    self.dyn_parameter_case = 0  # 0 - parameter in input,  1 - parameter in output

    self.build_NN(seed=0)

  # these are wrappers that call the tf.functions defined above, for efficiency
  def loss_total(self, dataset, full_loss=True):
    # params = self.nn_params
    # return self.loss_total_(dataset, full_loss, params)
    x = dataset[0]
    y = dataset[1]
    if self.dyn_parameter_case==0:
      u = x[:, -self.n_params:]
      state = y
    elif self.dyn_parameter_case==1:
      u = y[:,-self.n_params:]
      state = y[:, :self.n_state]
    return loss_total(self.model, x, u, state, y, self.dynamics_function, self.var_prior, self.var_data, self.var_phys, full_loss)
  
  def grad_loss(self, dataset, full_loss=True):
    x = dataset[0]
    y = dataset[1]
    if self.dyn_parameter_case==0:
      u = x[:, -self.n_params:]
      state = y
    elif self.dyn_parameter_case==1:
      u = y[:,-self.n_params:]
      state = y[:, :self.n_state]
    grad_thetas = grad_loss_total(self.model, x, u, state, y, self.dynamics_function, self.var_prior, self.var_data, self.var_phys, full_loss)
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
estimator_model.skip_phys_loss = False

# n_samples = 500
# xr = tf.random.normal(shape=(n_samples,estimator_model.n_inputs))
# yr = tf.random.normal(shape=(n_samples,estimator_model.n_outputs))
# dset = (xr, yr)
# y = estimator_model.loss_total(dset)
# J = estimator_model.grad_loss(dset)

# hmc = my_HMC(estimator_model, par_method, data_batch=dset, debug_flag=True)

# theta_list = []
# start = time.time()
# for _ in range(par_method['epochs']):
#     theta1 = hmc.sample_theta(estimator_model.nn_params)
#     theta_list.append(theta1)
# end = time.time()
# print("time elapsed: ", end - start)

def flatten_tensor_list(tensor_list):
  flattened_tensors = list(map(lambda t : tf.reshape(t, (-1)), tensor_list))
  return tf.concat(flattened_tensors, axis=0)

def theta_statistics(theta_list):
    theta_sum = theta_list[0]
    for th in theta_list[1:]:
        theta_sum += th
    theta_mean = theta_sum/len(theta_list)
    theta_var = (theta_list[0] - theta_mean)**2
    for th in theta_list[1:]:
        theta_var += (th - theta_mean)**2
    theta_var = theta_var/(len(theta_list)-1)
    return theta_mean, theta_var


# structure that contains tracklet information
class TrackletData:
   def __init__(self, T, X, Y, u):
      self.T = np.asarray(T)  # time array (1D array of length n_obs_sequence)
      self.X = np.asarray(X)  # state array (n_obs_sequence X n_states)
      self.Y = np.asarray(Y)  # observation array (n_obs_sequence X n_obs)
      self.u = np.asarray(u)  # dynamical model parameter (1D array of length n_params)

      self.N = len(T)  # if we want we can check everything else has the correct dimensions

def TrackletData2TrainingData(trdt, tArrayTrain):
  model_input_list = []
  model_output_list = []
  for t in tArrayTrain:
     xt = trajectory_propagator(trdt.X[0], t, trdt.u)
     model_input_list.append(np.concatenate(
        [[t], np.reshape(trdt.Y, (-1)), trdt.u]
     ))
     model_output_list.append(xt[0,:])
  return np.stack(model_input_list), np.stack(model_output_list)

# def TrackletData2TrainingData(trdt, n_obs_sequence):
#     model_input_list = []
#     model_output_list = []
#     for i in range(trdt.N-n_obs_sequence+1):  # for each sequence of observations in a tracklet
#         j = i+n_obs_sequence-1
#         model_input_list.append(np.concatenate(
#            [[trdt.T[j]], np.reshape(trdt.Y[i:(i+n_obs_sequence),:], (-1)), trdt.u]
#         ))
#         model_output_list.append(trdt.X[j, :])
#     return np.stack(model_input_list), np.stack(model_output_list)

def measurement_function(x):
   return x[:,[0]]

np.random.seed(0)
tArrayObs = np.asarray(range(estimator_model.n_obs_sequence), dtype=np.float64)
tArrayTrain = np.linspace(0, estimator_model.n_obs_sequence-1, estimator_model.n_obs_sequence*4)
n_samples = 1000
training_fraction = 0.7

k_sample = np.random.uniform(0.5, 2.0, (n_samples, estimator_model.n_params))
X_sample = np.random.uniform(-1.0, 1.0, (n_samples, estimator_model.n_state))
mdin_list = []
mdout_list = []
print('generating training data...')
tracklet_list = []
for i in range(n_samples):
    print(f'{i}/{n_samples}', end='\r')
    X_traj = trajectory_propagator(X_sample[i,:], tArrayObs, k_sample[i,:])  # , k
    Y_traj = measurement_function(X_traj)
    trdt = TrackletData(tArrayObs, X_traj, Y_traj, k_sample[i,:])
    mdin, mdout = TrackletData2TrainingData(trdt, tArrayTrain)
    mdin_list.append(mdin)
    mdout_list.append(mdout)
    tracklet_list.append(trdt)
print(f'{n_samples}/{n_samples}')

xdset = tf.constant(np.concatenate(mdin_list, axis=0), dtype=tf.float32)
ydset = tf.constant(np.concatenate(mdout_list, axis=0), dtype=tf.float32)

dset = (xdset, ydset)

n_training = int(np.floor(xdset.shape[0]*training_fraction))
dset_training = (xdset[:n_training], ydset[:n_training])

hmc = my_HMC(estimator_model, par_method, data_batch=dset, debug_flag=True)

# train model on actual simulated data
theta_list = []
start = time.time()
for _ in range(par_method['epochs']):
    theta1 = hmc.sample_theta(estimator_model.nn_params)
    theta_list.append(theta1)
end = time.time()
t_elapsed = end-start
print("time elapsed: ", t_elapsed)

theta_mean_list = []
theta_var_list = []
NW = 10
for k in range(len(theta_list)-NW):
    theta_sublist = theta_list[k:(k+NW)]
    theta_mean, theta_var = theta_statistics(theta_sublist)
    theta_mean_list.append(flatten_tensor_list(theta_mean.values).numpy())
    theta_var_list.append(flatten_tensor_list(theta_var.values).numpy())
theta_mean_list = np.stack(theta_mean_list)
theta_var_list = np.stack(theta_var_list)

theta_list_np = np.stack([flatten_tensor_list(th.values).numpy() for th in theta_list])


plt.figure()
plt.plot(theta_list_np[:,0])

plt.figure()
plt.plot(theta_mean_list[:,0])
# plt.plot(theta_mean_list[:,1000])
# plt.plot(theta_mean_list[:,2000])

plt.figure()
plt.plot(np.sqrt(theta_var_list[:,0]))




## test the method:
#simulate a trajectory, obtain some observations (no noise for now) and test the prediction of the model
th_test = theta_list[-1]  # take a single sample for now, but multiple should be used in the future
estimator_model.nn_params = th_test

i_test_case = 0
t_array_test = np.linspace(0.0, 2.0, 200)  # to test ability to interpolate, higher number of sample points. to test ability to extrapolate, going out of bounds a bit
x_test = xdset.numpy()[[i_test_case],:]
trdt = tracklet_list[i_test_case]
state_output = np.zeros((len(t_array_test), estimator_model.n_state))
X_real = trajectory_propagator(trdt.X[0], t_array_test, trdt.u)

for i in range(len(t_array_test)):
  # change the time input while keeping the others the same. code looks overcomplicated because it seems it's not possible to assign to a tensor element
  x_test[0,0] = t_array_test[i]
  state_output[i,:] = estimator_model.model(x_test).numpy()

plt.figure()
plt.plot(t_array_test, state_output[:,0], label="predicted")
plt.plot(t_array_test, X_real[:,0], label="ground truth")

plt.figure()
plt.plot(hmc.hamiltonians[10:])

plt.show()


# save results
fname = time.strftime("Results/%Y%m%d-%H%M%S.pckl")
par_nn_ = copy(par_nn)
par_nn_["dynamics_function"] = []  # removing function, which can't be saved by pickle
obj = [
   dset, n_training, par_nn_, par_method, estimator_model.nn_params.values, theta_list, t_elapsed
]
with open(fname, 'wb') as f:
  pickle.dump(obj, f)