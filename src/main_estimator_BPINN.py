import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

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
  with tf.GradientTape(persistent=True) as tape:  # persistent=True, watch_accessed_variables=False
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

par_method = {'epochs': 10, # number of samples
              'burn_in': 0, 
              'HMC_skip': 1,  # UNUSED
              'HMC_L': 10,  # number of leapfrog steps per sample of theta
              'HMC_dt': 1e-3,  # time step length for leapfrog steps
              'HMC_eta': 0.5*1e-3  # mass of artificial Hamiltonian system
              }
  
# making HMC_eta smaller improved acceptance rate, but I would guess has consequences in terms of effectively sampling the distribution

par_nn = {'n_state': 2, # number of state variables
          'n_obs': 1, # number of variables per observation
          'n_obs_sequence': 3, # number of observations in a sequence fed to the model
          'n_params': 0, # number of parameters to the dynamics
          'n_neurons': [50, 50], # number of neurons in each layer
          'activation_function': 'swish',  # must be something tf.keras.layers.Dense can accept as activation
          'dynamics_function': lambda x,t,u : dynamics(x,t,1.0),  # 
          'stddev': 1,  # stddev of initialisation of NN
          'std_prior': 1.0,  # standard deviation of prior
          'std_phys': [1.0, 1.0],  # standard deviation of state information, in units of state
          'std_data': [1.0, 1.0],  # standard deviation of state derivative information
          }  


# wrappers to try to use @tf.function graph optimisation
# @tf.function
# def loss_total_(model, dataset, nn_params):
#    return model.loss_total_(dataset, nn_params)

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

    self.var_prior = np.asarray(par_nn['std_prior'])**2
    self.var_phys =  np.asarray(par_nn['std_phys'])**2
    self.var_data =  np.asarray(par_nn['std_data'])**2

    self.dyn_parameter_case = 0  # 0 - parameter in input,  1 - parameter in output

    self.build_NN(seed=0)

  # loss functions are equal to -log( P(D|theta)) ) apart from constant terms, which since theyre gaussian, means they equal (1/2)*((x-mu)/sigma)^2
  def data_loss(self, x, y):
     ypred = self.model(x)
     x_mu_diff = tf.math.reduce_sum(tf.square(y-ypred), axis=0)
     return tf.math.reduce_sum(x_mu_diff/(self.var_data))/2.0

  def phys_loss(self, x):
    f_nn = time_deriv(self.model, x)

    y = self.model(x)
    if self.dyn_parameter_case==0:
       u = x[:, -self.n_params:]
       state = y
    elif self.dyn_parameter_case==1:
       u = y[:,-self.n_params:]
       state = y[:, :self.n_state]
    t = x[:, 0]
    f_known = self.dynamics_function(state, t, u)
    x_mu_diff = tf.math.reduce_sum(tf.square(f_nn-f_known), axis=0)
    # return f_nn, f_known
    return tf.math.reduce_sum(x_mu_diff/(self.var_phys))/2.0
  
  def prior_loss(self, params):  # see comment in loss_total_ (with underscore at the end)
    return params.ssum()/(2.0*self.var_prior)

  def loss_total(self, dataset, full_loss=True):
    params = self.nn_params
    return self.loss_total_(dataset, full_loss, params)
  
  # helper function because tf.functions can't access nn_params for some reason
  # this is the only way to build the computational graph with tf.function
  # however, I later found out that making this a tf.function isn't efficient. try it if you like:
  # @tf.function
  def loss_total_(self, dataset, full_loss, params):
    L = self.prior_loss(params)
    L += self.data_loss(dataset[0], dataset[1])
    if full_loss:
       L += self.phys_loss(dataset[0])
    return L
  
  # def loss_total(self, dataset, full_loss=True):
  #   # params = self.nn_params
  #   L1 = self.prior_loss()
  #   L2 = self.data_loss(dataset[0], dataset[1])
  #   if full_loss:
  #      L3 = self.phys_loss(dataset[0])
  #   else:
  #      L3 = 0
  #   return L1 + L2 + L3

  # idential to the one present in LossNN
  def grad_loss(self, dataset, full_loss=True):
    # fl = lambda x : self.loss_total(x)
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



# def lo

estimator_model = EstimatorNN(par_nn)
estimator_model.skip_phys_loss = False

n_samples = 100
xr = tf.random.normal(shape=(n_samples,estimator_model.n_inputs))
yr = tf.random.normal(shape=(n_samples,estimator_model.n_outputs))
dset = (xr, yr)
y = estimator_model.loss_total(dset)
J = estimator_model.grad_loss(dset)

hmc = my_HMC(estimator_model, par_method, data_batch=dset, debug_flag=True)

theta_list = []
start = time.time()
for _ in range(par_method['epochs']):
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
        theta_var = (th - theta_mean)**2
    theta_var = theta_var/(len(theta_list)-1)
    return theta_mean, theta_var

theta_mean_list = []
theta_var_list = []
NW = 3
for k in range(len(theta_list)-NW):
    theta_sublist = theta_list[k:(k+NW)]
    theta_mean, theta_var = theta_statistics(theta_sublist)
    theta_mean_list.append(flatten_tensor_list(theta_mean.values).numpy())
    theta_var_list.append(flatten_tensor_list(theta_var.values).numpy())
theta_mean_list = np.stack(theta_mean_list)
theta_var_list = np.stack(theta_var_list)

theta_list_np = np.stack([flatten_tensor_list(th.values).numpy() for th in theta_list])


# plt.figure()
# plt.plot(theta_list_np[:,0])

# plt.figure()
# plt.plot(theta_mean_list[:,0])
# # plt.plot(theta_mean_list[:,1000])
# # plt.plot(theta_mean_list[:,2000])

# plt.figure()
# plt.plot(theta_var_list[:,0])


# structure that contains tracklet information
class TrackletData:
   def __init__(self, T, X, Y, u):
      self.T = np.asarray(T)  # time array (1D array of length n_obs_sequence)
      self.X = np.asarray(X)  # state array (n_obs_sequence X n_states)
      self.Y = np.asarray(Y)  # observation array (n_obs_sequence X n_obs)
      self.u = np.asarray(u)  # dynamical model parameter (1D array of length n_params)

      self.N = len(T)  # if we want we can check everything else has the correct dimensions

def TrackletData2TrainingData(trdt, n_obs_sequence):
    model_input_list = []
    model_output_list = []
    for i in range(trdt.N-n_obs_sequence+1):
        j = i+n_obs_sequence-1
        model_input_list.append(np.concatenate(
           [[trdt.T[j]], np.reshape(trdt.Y, (-1)), trdt.u]
        ))
        model_output_list.append(trdt.X[j, :])
    return np.stack(model_input_list), np.stack(model_output_list)

def measurement_function(x):
   return x[:,0]

np.random.seed(0)
t_array = np.linspace(0.0, 10.0, 11)
n_samples = 100
k = [1.0]
X_sample = np.random.uniform(-1.0, 1.0, (n_samples, estimator_model.n_state))
mdin_list = []
mdout_list = []
print('generating training data...')
for i in range(n_samples):
    print(f'{i}/{n_samples}', end='\r')
    X_traj = trajectory_propagator(X_sample[i,:], t_array, k)
    Y_traj = measurement_function(X_traj)
    trdt = TrackletData(t_array, X_traj, Y_traj, k)
    mdin, mdout = TrackletData2TrainingData(trdt, estimator_model.n_obs_sequence)
    mdin_list.append(mdin)
    mdout_list.append(mdout)
print(f'{n_samples}/{n_samples}')


plt.show()