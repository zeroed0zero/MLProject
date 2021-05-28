from Data.random_models import getRandomArray
from Src.common import *
from pandas import read_csv
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def perceptron(X, synapses, targets, epochs, beta, activation_func):
    have_changed = True # synapses have changed
    while epochs != 0 and have_changed == True:
        epochs -= 1
        have_changed = False
        for i in range(len(X)):
            output = activation_func(np.matmul(synapses, X[i]))
            if not np.array_equal(output, targets[i]):
                synapses = synapses + (beta * (targets[i] - output)) * X[i]
                have_changed = True
    return synapses

def perceptron_outputs(X, synapses):
    return [np.matmul(synapses, p) for p in X]

def perceptron_outputs_act(X, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in X])

def perceptron_train_2D(data_file, beta, epochs):
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

    old_synapses = []
    for i in range(epochs):
        x,y = get_separation_line(synapses)
        classifications = perceptron_outputs_act(X_biased, synapses, lambda v: 0 if v < 0 else 1)
        update_line_figure(data, classifications, ax2, x, y)
        update_class_figure_2D(data, classifications, ax3)
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(X_biased, synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        plt.pause(0.7)

def perceptron_recall_2D(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    ax = figure.add_subplot(gs[:,:])
    ax.legend(handles=[mpatches.Patch(color='blue', label='Actual targets'),
                       mpatches.Patch(color='red', label='Predicted targets')])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 3)
    data = read_csv(data_file, header=None)[[0,1,2]].values
    X = data[:,:2]
    targets = data[:,2]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))[:,:3]
    X_train, X_test, y_train, y_test = \
        train_test_split(X_biased, targets, test_size=.4, random_state=42)

    synapses = perceptron(X_train, synapses, targets, epochs, beta, lambda v: 0 if v < 0 else 1)

    y_test_predict = perceptron_outputs_act(X_test, synapses, lambda v: 0 if v < 0 else 1)
    ax.plot(list(range(len(y_test))), y_test, 'b.')
    ax.plot(list(range(len(y_test_predict))), y_test_predict, 'ro', fillstyle='none')

def perceptron_train_3D(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0], projection='3d')
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1], projection='3d')
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 4)
    data = read_csv(data_file, header=None)
    X = data[[0,1,2,3]].values
    targets = X[:,3]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))[:,:4]

    plot_prototypes_3D(X, ax1)

    old_synapses = []
    for i in range(epochs):
        x,y,z = get_separation_plane(synapses)
        classifications = perceptron_outputs_act(X_biased, synapses, lambda v: 0 if v < 0 else 1)
        update_plane_figure(X, ax2, classifications, x, y, z)
        update_class_figure_3D(X, classifications, ax3)
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(X_biased, synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        plt.pause(0.7)

def perceptron_recall_3D(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    ax = figure.add_subplot(gs[:,:])
    ax.legend(handles=[mpatches.Patch(color='blue', label='Actual targets'),
                       mpatches.Patch(color='red', label='Predicted targets')])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 4)
    data = read_csv(data_file, header=None)[[0,1,2,3]].values
    X = data[:,:3]
    targets = data[:,3]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))[:,:4]

    X_train, X_test, y_train, y_test = \
        train_test_split(X_biased, targets, test_size=.4, random_state=42)

    synapses = perceptron(X_train, synapses, targets, epochs, beta, lambda v: 0 if v < 0 else 1)

    y_test_predict = perceptron_outputs_act(X_test, synapses, lambda v: 0 if v < 0 else 1)
    ax.plot(list(range(len(y_test))), y_test, 'b.')
    ax.plot(list(range(len(y_test_predict))), y_test_predict, 'ro', fillstyle='none')

def perceptron_flowers_train(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1'),
                            mpatches.Patch(color='green', label='Class 2')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()
    
    synapses = getRandomArray(-0.5, 0.5, 3)
    data = read_csv(data_file, header=None)
    X = data[[0,2,4]].values
    for i in range(len(X)):
        if X[i][2] == 'Iris-setosa': X[i][2] = 1
        elif X[i][2] == 'Iris-versicolor': X[i][2] = 2
        elif X[i][2] == 'Iris-virginica': X[i][2] = 3
    # same as X, the first class is 0 and the last 2
    # needed for plotting the X and the class figure
    X_0_2 = X.copy()
    for i in range(len(X_0_2)): X_0_2[i][2] -= 1
    targets = X[:,2]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))[:,:3]
    
    targets_setosa = targets.copy()
    for i in range(len(targets_setosa)):
        if targets_setosa[i] in [2,3]: targets_setosa[i] = 0
    targets_versicolor = targets.copy()
    for i in range(len(targets_versicolor)):
        if targets_versicolor[i] in [1,3]: targets_versicolor[i] = 0
    targets_virginica = targets.copy()
    for i in range(len(targets_virginica)):
        if targets_virginica[i] in [1,2]: targets_virginica[i] = 0
    
    synapses_setosa = synapses.copy()
    synapses_versicolor = synapses.copy()
    synapses_virginica = synapses.copy()
    classes = [0, 1, 2]
    plot_prototypes_2D(X_0_2, ax1)
    for i in range(epochs):
        synapses_setosa = perceptron(X_biased, synapses_setosa, targets_setosa, 1, beta, lambda v: 0 if v < 0 else 1)
        synapses_versicolor = perceptron(X_biased, synapses_versicolor, targets_versicolor, 1, beta, lambda v: 0 if v < 0 else 2)
        synapses_virginica = perceptron(X_biased, synapses_virginica, targets_virginica, 1, beta, lambda v: 0 if v < 0 else 3)
        p1 = perceptron_outputs(X_biased, synapses_setosa)
        p2 = perceptron_outputs(X_biased, synapses_versicolor)
        p3 = perceptron_outputs(X_biased, synapses_virginica)
        pred_classes = []
        for i in range(len(p1)):
            pred_classes.append(classes[np.argmax([p1[i], p2[i], p3[i]])])
        update_pred_figure(X, ax2, pred_classes)

        update_class_figure(X_0_2, ax3, pred_classes, 2)
        plt.pause(.000001)

def perceptron_flowers_recall(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    ax = figure.add_subplot(gs[:,:])
    ax.legend(handles=[mpatches.Patch(color='blue', label='Actual targets'),
                            mpatches.Patch(color='red', label='Predicted targets')])
    figure.show()
    
    synapses = getRandomArray(-0.5, 0.5, 3)
    data = read_csv(data_file, header=None)[[0,2,4]].values
    for i in range(len(data)):
        if data[i][2] == 'Iris-setosa': data[i][2] = 1
        elif data[i][2] == 'Iris-versicolor': data[i][2] = 2
        elif data[i][2] == 'Iris-virginica': data[i][2] = 3
    X = data[:,:2]
    targets = data[:,2]
    rows, _ = X.shape
    bias = np.ones((rows, 1)) * -1
    X_biased = np.hstack((bias, X))[:,:3]
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X_biased, targets, test_size=.4, random_state=42)
    
    targets_setosa = y_train.copy()
    for i in range(len(targets_setosa)):
        if targets_setosa[i] in [2,3]: targets_setosa[i] = 0
    targets_versicolor = y_train.copy()
    for i in range(len(targets_versicolor)):
        if targets_versicolor[i] in [1,3]: targets_versicolor[i] = 0
    targets_virginica = y_train.copy()
    for i in range(len(targets_virginica)):
        if targets_virginica[i] in [1,2]: targets_virginica[i] = 0

    synapses_setosa = perceptron(X_train, synapses, targets_setosa, epochs, beta, lambda v: 0 if v < 0 else 1)
    synapses_versicolor = perceptron(X_train, synapses, targets_versicolor, epochs, beta, lambda v: 0 if v < 0 else 2)
    synapses_virginica = perceptron(X_train, synapses, targets_virginica, epochs, beta, lambda v: 0 if v < 0 else 3)
    p1 = perceptron_outputs(X_test, synapses_setosa)
    p2 = perceptron_outputs(X_test, synapses_versicolor)
    p3 = perceptron_outputs(X_test, synapses_virginica)
    classes = [0, 1, 2]
    y_test_predict = []
    for i in range(len(p1)):
        y_test_predict.append(classes[np.argmax([p1[i], p2[i], p3[i]])])

    ax.plot(list(range(len(y_test))), y_test, 'b.')
    ax.plot(list(range(len(y_test_predict))), y_test_predict, 'ro', fillstyle='none')
