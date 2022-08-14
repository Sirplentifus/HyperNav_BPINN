import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def plot_losses(path_plot, losses):
    """
    Plot log(losses)
    """
    plt.figure()
    plt.plot(np.log(losses['Loss']), 'k--', lw=2.5, alpha=1.0, label = 'Loss_Total')
    for key,value in losses.items():
        if not key == 'Loss':
            plt.plot(np.log(value), lw=1.0, alpha=0.7, label = key)
    plt.xlabel('epochs')
    plt.ylabel('LogLoss')
    plt.legend(prop={'size': 9})
    path = os.path.join(path_plot,"loss.png")
    plt.savefig(path, bbox_inches= 'tight')

def load_losses(path_result):
    losses = dict()
    for loss_filename in os.listdir(path_result):
        if loss_filename[-4:] == ".csv":
            with open(os.path.join(path_result,loss_filename)) as loss_file:
                csvreader = csv.reader(loss_file)
                loss_list = list()
                for loss_value in csvreader:
                    loss_list.append(loss_value)
            losses[loss_filename[:-4]] = np.array(loss_list, dtype='float32')
    return losses

def plot_confidence(path_plot, datasets_class, functions):

    inputs, u_true, f_true = datasets_class.dom_data
    u_points, u_values, _  = datasets_class.exact_data_noise

    u = (u_true, functions['u_NN'], functions['u_std'])
    u_fit = (u_points, u_values)
    f = (f_true, functions['f_NN'], functions['f_std'])

    plot_1D(inputs[:,0], u, 'Confidence interval for u(x)', label = ('x','u'), fit = u_fit)
    save_plot(path_plot, 'u_confidence.png')
    plot_1D(inputs[:,0], f, 'Confidence interval for f(x)', label = ('x','f'))
    save_plot(path_plot, 'f_confidence.png')


def plot_1D(x, func, title, label = ("",""), fit = None):
    
    idx = np.argsort(x)
    x = x[idx]
    func = [i[idx] for i in func]

    plt.figure()
    plt.plot(x, func[0], 'r-', label='true')
    plt.plot(x, func[1], 'b--', label='mean')
    plt.plot(x, func[1] - func[2], 'g--', label='mean-std')
    plt.plot(x, func[1] + func[2], 'g--', label='mean+std')
    if fit is not None:
        plt.plot(fit[0], fit[1], 'r*')

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(prop={'size': 9})
    plt.title(title)


def plot_nn_samples(path_plot, datasets_class, functions_all, method):
    inputs, u_true, f_true = datasets_class.dom_data
    u_points, u_values, _  = datasets_class.exact_data_noise

    u = (u_true, functions_all['u_NN'])
    u_fit = (u_points, u_values)
    f = (f_true, functions_all['f_NN'])

    plot_1Dall(inputs[:,0], u, method, label = ('x','u'), fit = u_fit)
    save_plot(path_plot, 'u_nn_samples.png')
    plot_1Dall(inputs[:,0], f, method, label = ('x','f'), fit = None)
    save_plot(path_plot, 'f_nn_samples.png')

def plot_1Dall(x, func, method, label = ("",""), fit = None):

    idx = np.argsort(x)
    x = x[idx]

    plt.figure()

    if(method == "SVGD"):
        for i in range(func[1].shape[1]):
            plt.plot(x, func[1][idx,i])
    elif(method == "HMC"):
        for i in range(func[1].shape[0]):
            plt.plot(x, func[1][i,idx,0], 'b-', markersize=0.01, alpha=0.01)
    plt.plot(x, func[0][idx], 'r-', label='true')
    if fit is not None:
        plt.plot(fit[0], fit[1], 'r*')

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(prop={'size': 9})
    if method == "SVGD":
        plt.title('Output ' + label[1] + ' of all networks')
    elif method == "HMC":
        plt.title('Samples from ' + label[1] + ' reconstructed distribution')

def save_plot(path_plot, title):
    path = os.path.join(path_plot, title)
    plt.savefig(path, bbox_inches = 'tight')
    
def show_plot():
    plt.show(block = True)

if __name__ == "__main__":
    my_path = '../../1D-laplace/trash'
    losses = load_losses(my_path)
    plot_losses(my_path, losses)
    show_plot()