import numpy as np
import tensorflow as tf

def trajectory_propagator(x0, tArray, k):
    # given initial state x0, array of time values tArray, and the dynamics parameters k, 
    # this outputs the solution to the ode:
    # d x[0] / dt = x[1]
    # d x[1] / dt = -k*x[0]

    r0 = x0[0]  # initial position
    v0 = x0[1]  # initial velocity

    A = np.sqrt(r0**2 + v0**2/k)  # amplitude
    w = np.sqrt(k)  # angular velocity

    s_phi = r0/A
    c_phi = v0/A/w
    phi = np.arctan2(s_phi, c_phi)

    r = A * np.sin(w*tArray + phi)
    v = A * w * np.sin(w*tArray + phi)

    # return np.vstack((r,v)).T
    return np.stack((r,v), axis=1)

def dynamics(xArray, tArray, k):
    # computes the derivative of the state from known (or assumed) dynamics
    # for the harmonic oscillator, the dynamics do not explicitly depend on
    # time, but in general the dynamics might depend on it, which is why
    # the argument tArray is here

    rdot = xArray[:,1]
    vdot = -k*xArray[:,0]
    return tf.stack((rdot,vdot), axis=1)
    # return np.vstack((rdot,vdot)).T

# an example
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    x0 = [2, 1]
    tArray = np.linspace(0, 1, 100)
    T = 1/2 # period
    w = 2*np.pi/T  # ang. vel.
    k = w**2
    xArray = trajectory_propagator(x0, tArray, k)

    xDotArray = dynamics(xArray, tArray, k)

    plt.plot(tArray, xArray[0,:], label="r")
    plt.plot(tArray, xArray[1,:], label="v")
    plt.plot(tArray, xDotArray[0,:], '--', label="r dot")
    plt.plot(tArray, xDotArray[1,:], '--', label="v dot")
    plt.ylabel('states')
    plt.xlabel('time')
    plt.legend()
    plt.show()