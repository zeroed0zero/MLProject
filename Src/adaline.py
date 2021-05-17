from Src.common import *
from Data.random_models import getRandomArray
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def mse(prototypes, synapses, targets):
    return 0.5 * sum([(targets[i] - np.matmul(synapses, prototypes[i]))**2 for i in range(len(prototypes))])

def adaline(prototypes, synapses, targets, epochs, beta, mse_limit):
    while epochs != 0 and mse(prototypes, synapses, targets) > mse_limit:
        print(mse(prototypes, synapses, targets))
        epochs -= 1
        for i in range(len(prototypes)):
            output = np.matmul(synapses, prototypes[i])
            synapses = synapses + (beta * (targets[i] - output)) * prototypes[i]
    return synapses

def adaline_train_2D(data_file, synapses, beta):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    data = read_csv(data_file, header=None)
    prototypes = data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))
    plot_prototypes_2D(prototypes, fig_ax1)
    x,y = get_separation_line(synapses)
    update_line_figure(prototypes, fig_ax2, x, y)
    update_class_figure_2D(prototypes, fig_ax3, x, y)
    old_synapses = []
    for i in range(1000):
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = adaline(prototypes_biased[:,:3], synapses, targets, 1, beta, .1)
        x,y = get_separation_line(synapses)
        update_line_figure(prototypes, fig_ax2, x, y)
        update_class_figure_2D(prototypes, fig_ax3, x, y)
        plt.pause(0.7)
