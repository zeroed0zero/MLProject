from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from Src.common import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def mlp_train_2D(train_data_file, synapses, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, 0])
    fig_ax4 = figure.add_subplot(gs[1, 1])
    figure.show()

    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:3]

    plot_prototypes_2D(prototypes, fig_ax1)
    clf = MLPClassifier(random_state=1, max_iter=epochs, hidden_layer_sizes=(256), activation='tanh', solver='adam', learning_rate_init=beta)
    clf.partial_fit(prototypes_biased, targets, [0,1])

    epoch_values = []
    mse_values = []
    for i in range(epochs):
        synapses = np.ndarray.flatten(clf.coefs_[0])
        classifications = clf.predict(prototypes_biased)
        epoch_values.append(i)
        mse_values.append(mean_squared_error(targets, classifications))
        clf.partial_fit(prototypes_biased, targets)
        x,y = get_separation_line(synapses)
        update_line_figure(prototypes, classifications, fig_ax2, x, y)
        update_class_figure_2D(prototypes, classifications, fig_ax3)
        update_mse_figure(mse_values, epoch_values, fig_ax4)
        plt.pause(0.00001)

def mlp_recall_2D(train_data_file, synapses, beta, epochs):
    print("Not implemented yet\n")
def mlp_train_3D(train_data_file, synapses, beta, epochs):
    print("Not implemented yet\n")
def mlp_recall_3D(train_data_file, synapses, beta, epochs):
    print("Not implemented yet\n")
