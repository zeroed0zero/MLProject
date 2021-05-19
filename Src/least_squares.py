from Src.common import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def least_squares(prototypes, synapses, targets, epochs, ls_limit):
    for i in range(epochs):
        synapses = 

def perceptron_outputs(prototypes, synapses):
    return [np.matmul(synapses, p) for p in prototypes]

def perceptron_outputs_act(prototypes, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in prototypes])

def perceptron_train_2D(train_data_file, synapses, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    train_data = read_csv(train_data_file, header=None)
    prototypes = data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))

    plot_prototypes_2D(prototypes, fig_ax1)

    for i in range(epochs):
        x,y = get_separation_line(synapses)
        classifications = perceptron_outputs_act(prototypes_biased[:,:3], synapses, lambda v: 0 if v < 0 else 1)
        update_line_figure(prototypes, classifications, fig_ax2, x, y)
        update_class_figure_2D(prototypes, classifications, fig_ax3)
        synapses = perceptron(prototypes_biased[:,:3], synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        plt.pause(0.7)
