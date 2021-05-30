from Data.random_models import *
from Src.common import *
from pandas import read_csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def least_squares(prototypes, synapses, targets):
    return np.matmul(np.linalg.pinv(prototypes), targets)

def lsq_outputs_act(X, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in X])

def lsq_train_2D(data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 3)
    data = read_csv(data_file, header=None)[[0,1,2]].values
    X = data[:,:2]
    targets = data[:,2]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))

    plot_prototypes_2D(data, ax1)
    synapses = least_squares(X_biased, synapses, targets)
    x,y = get_separation_line(synapses)
    classifications = lsq_outputs_act(X_biased, synapses, lambda v: 0 if v < 0 else 1)
    update_line_figure(data, classifications, ax2, x, y)
    update_class_figure_2D(data, classifications, ax3)

def lsq_recall_2D(data_file):
    print("Not implemented yet\n")
def lsq_train_3D(data_file):
    print("Not implemented yet\n")
def lsq_recall_3D(data_file):
    print("Not implemented yet\n")
def lsq_flowers_train(data_file):
    print("Not implemented yet\n")
def lsq_flowers_recall(data_file):
    print("Not implemented yet\n")
