from Src.common import *
from Data.random_models import getRandomArray
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def perceptron(prototypes, synapses, targets, epochs, beta, activation_func):
    have_changed = True # synapses have changed
    while epochs != 0 and have_changed == True:
        epochs -= 1
        have_changed = False
        for i in range(len(prototypes)):
            old_synapses = np.copy(synapses)
            output = activation_func(np.matmul(synapses, prototypes[i]))
            if not np.array_equal(output, targets[i]):
                synapses = old_synapses + (beta * (targets[i] - output)) * prototypes[i]
                have_changed = True
    return synapses

def perceptron_outputs(prototypes, synapses):
    return [np.matmul(synapses, p) for p in prototypes]

def perceptron_recall(prototypes, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in prototypes])

def perceptron_train_3D(data_file, synapses, beta):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0], projection='3d')
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1], projection='3d')
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    data = read_csv(data_file, header=None)
    prototypes = data[[0,1,2,3]].values

    targets = prototypes[:,3]

    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))

    plot_prototypes_3D(prototypes, fig_ax1)

    x,y,z = get_separation_plane(synapses)
    update_plane_figure(prototypes, fig_ax2, x, y, z)
    update_class_figure_3D(prototypes, fig_ax3, x, y, z)

    old_synapses = []
    for i in range(1000):
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased[:,:4], synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)

        x,y,z = get_separation_plane(synapses)
        update_plane_figure(prototypes, fig_ax2, x, y, z)
        update_class_figure_3D(prototypes, fig_ax3, x, y, z)
        plt.pause(0.7)
    

def perceptron_recall_3D(data_file, synapses, beta):
    return 1

def perceptron_recall_2D(data_file, synapses, beta):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    axes = figure.add_subplot(gs[:,:])
    axes.legend(handles=[mpatches.Patch(color='green', label='Correct predictions'),
                            mpatches.Patch(color='red', label='Incorrect predictions')])
    figure.show()
    data = read_csv(data_file, header=None)
    prototypes = data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))
    synapses = perceptron(prototypes_biased[:,:3], synapses, targets, 200, beta, lambda v: 0 if v < 0 else 1)
    x,y = get_separation_line(synapses)
    predict_targets = perceptron_recall(prototypes_biased[:,:3], synapses, lambda v: 0 if v < 0 else 1)

    predicted_t = []
    for i in range(len(targets)):
        if targets[i] == predict_targets[i]:
            predicted_t.append(True)
        else:
            predicted_t.append(False)
    recall_figure_2D(prototypes, predicted_t, axes)
    axes.axline(x, y, color='00')

def perceptron_train_2D(data_file, synapses, beta):
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
        synapses = perceptron(prototypes_biased[:,:3], synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        x,y = get_separation_line(synapses)
        update_line_figure(prototypes, fig_ax2, x, y)
        update_class_figure_2D(prototypes, fig_ax3, x, y)
        plt.pause(0.7)

def perceptron_flowers_training(data_file, synapses, beta):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1'),
                            mpatches.Patch(color='green', label='Class 2')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()
    
    data = read_csv(data_file, header=None)
    prototypes = data[[0,2,4]].values
    for i in range(len(prototypes)):
        if prototypes[i][2] == 'Iris-setosa': prototypes[i][2] = 1
        elif prototypes[i][2] == 'Iris-versicolor': prototypes[i][2] = 2
        elif prototypes[i][2] == 'Iris-virginica': prototypes[i][2] = 3
    # same as prototypes, the first class is 0 and the last 2
    # needed for plotting the prototypes and the class figure
    prototypes_0_2 = prototypes.copy()
    for i in range(len(prototypes_0_2)): prototypes_0_2[i][2] -= 1
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:3]
    
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
    plot_prototypes_2D(prototypes_0_2, fig_ax1)
    for i in range(10000):
        synapses_setosa = perceptron(prototypes_biased, synapses_setosa, targets_setosa, 1, beta, lambda v: 0 if v < 0 else 1)
        synapses_versicolor = perceptron(prototypes_biased, synapses_versicolor, targets_versicolor, 1, beta, lambda v: 0 if v < 0 else 2)
        synapses_virginica = perceptron(prototypes_biased, synapses_virginica, targets_virginica, 1, beta, lambda v: 0 if v < 0 else 3)
        p1 = perceptron_outputs(prototypes_biased, synapses_setosa)
        p2 = perceptron_outputs(prototypes_biased, synapses_versicolor)
        p3 = perceptron_outputs(prototypes_biased, synapses_virginica)
        pred_classes = []
        for i in range(len(p1)):
            pred_classes.append(classes[np.argmax([p1[i], p2[i], p3[i]])])
        update_pred_figure(prototypes, fig_ax2, pred_classes)

        update_class_figure(prototypes_0_2, fig_ax3, pred_classes, 2)
        plt.pause(.000001)

def perceptron_flowers_recall(data_file, synapses, beta):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    fig_ax1 = figure.add_subplot(gs[:,:])
    fig_ax1.legend(handles=[mpatches.Patch(color='green', label='Correct predictions'),
                            mpatches.Patch(color='red', label='Incorrect predictions')])
    figure.show()
    
    data = read_csv(data_file, header=None)
    prototypes = data[[0,2,4]].values
    for i in range(len(prototypes)):
        if prototypes[i][2] == 'Iris-setosa': prototypes[i][2] = 1
        elif prototypes[i][2] == 'Iris-versicolor': prototypes[i][2] = 2
        elif prototypes[i][2] == 'Iris-virginica': prototypes[i][2] = 3
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:3]
    
    targets_setosa = targets.copy()
    for i in range(len(targets_setosa)):
        if targets_setosa[i] in [2,3]: targets_setosa[i] = 0
    targets_versicolor = targets.copy()
    for i in range(len(targets_versicolor)):
        if targets_versicolor[i] in [1,3]: targets_versicolor[i] = 0
    targets_virginica = targets.copy()
    for i in range(len(targets_virginica)):
        if targets_virginica[i] in [1,2]: targets_virginica[i] = 0

    synapses_setosa = perceptron(prototypes_biased, synapses, targets_setosa, 10000, beta, lambda v: 0 if v < 0 else 1)
    synapses_versicolor = perceptron(prototypes_biased, synapses, targets_versicolor, 10000, beta, lambda v: 0 if v < 0 else 2)
    synapses_virginica = perceptron(prototypes_biased, synapses, targets_virginica, 10000, beta, lambda v: 0 if v < 0 else 3)
    p1 = perceptron_outputs(prototypes_biased, synapses_setosa)
    p2 = perceptron_outputs(prototypes_biased, synapses_versicolor)
    p3 = perceptron_outputs(prototypes_biased, synapses_virginica)
    classes = [0, 1, 2]
    pred_classes = []
    for i in range(len(p1)):
        pred_classes.append(classes[np.argmax([p1[i], p2[i], p3[i]])])

    # same as prototypes, the first class is 0 and the last 2
    # needed for plotting the prototypes and the class figure
    prototypes_0_2 = prototypes.copy()
    for i in range(len(prototypes_0_2)): prototypes_0_2[i][2] -= 1
    predicted_t = []
    for i in range(len(prototypes_0_2)):
        predicted_t.append(prototypes_0_2[i,2] == pred_classes[i])
    recall_figure_2D(prototypes_0_2, predicted_t, fig_ax1)
