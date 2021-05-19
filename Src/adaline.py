from Src.common import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def mse(prototypes, synapses, targets):
    return 0.5 * sum([(targets[i] - np.matmul(synapses, prototypes[i]))**2 for i in range(len(prototypes))])

def adaline(prototypes, synapses, targets, epochs, beta, mse_limit):
    while epochs != 0 and mse(prototypes, synapses, targets) > mse_limit:
        epochs -= 1
        for i in range(len(prototypes)):
            output = np.matmul(synapses, prototypes[i])
            synapses = synapses + (beta * (targets[i] - output)) * prototypes[i]
    return synapses

def adaline_outputs(prototypes, synapses):
    return [np.matmul(synapses, p) for p in prototypes]

def adaline_outputs_act(prototypes, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in prototypes])

def adaline_train_2D(data_file, synapses, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, 0])
    fig_ax4 = figure.add_subplot(gs[1, 1])
    figure.show()

    data = read_csv(data_file, header=None)
    prototypes = data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:3]

    plot_prototypes_2D(prototypes, fig_ax1)

    epoch_values = []
    mse_values = []

    for i in range(epochs):
        epoch_values.append(i)
        mse_values.append(mse(prototypes, synapses, targets))
        x,y = get_separation_line(synapses)
        classifications = adaline_outputs(prototypes_biased, synapses)
        update_line_figure_adaline(prototypes, classifications, fig_ax2, x, y)
        update_class_figure_2D(prototypes, classifications, fig_ax3)
        update_mse_figure(mse_values, epoch_values, fig_ax4)
        synapses = adaline(prototypes_biased, synapses, targets, 1, beta, .1)
        plt.pause(0.01)
    #from mlxtend.plotting import plot_decision_regions
    #from mlxtend.classifier import Adaline
    #import matplotlib.pyplot as plt
    #data = read_csv(data_file, header=None)
    #X = data[[0,1]].values
    #y = np.ndarray.flatten(data[[2]].values)
    #X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    #X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    #ada = Adaline(epochs=100, 
    #              eta=0.0001, 
    #              minibatches=None, print_progress=3)
    #ada.fit(X, y)
    #plot_decision_regions(X, y, clf=ada)
    #plt.show() 
