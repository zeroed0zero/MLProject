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
            output = activation_func(np.matmul(synapses, prototypes[i]))
            if not np.array_equal(output, targets[i]):
                synapses = synapses + (beta * (targets[i] - output)) * prototypes[i]
                have_changed = True
    return synapses

def perceptron_outputs(prototypes, synapses):
    return [np.matmul(synapses, p) for p in prototypes]

def perceptron_outputs_act(prototypes, synapses, activation_func):
    return np.array([activation_func(np.matmul(synapses, p)) for p in prototypes])

def perceptron_train_2D(train_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 3)
    train_data = read_csv(train_data_file, header=None)[[0,1,2]].values
    prototypes = train_data[:,:2]
    targets = train_data[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))

    plot_prototypes_2D(train_data, fig_ax1)

    old_synapses = []
    for i in range(epochs):
        x,y = get_separation_line(synapses)
        classifications = perceptron_outputs_act(prototypes_biased, synapses, lambda v: 0 if v < 0 else 1)
        update_line_figure(train_data, classifications, fig_ax2, x, y)
        update_class_figure_2D(train_data, classifications, fig_ax3)
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased, synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        plt.pause(0.7)

def perceptron_recall_2D(train_data_file, recall_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    axes = figure.add_subplot(gs[:,:])
    axes.legend(handles=[mpatches.Patch(color='green', label='Correct predictions'),
                         mpatches.Patch(color='red', label='Incorrect predictions')])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 3)
    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,1,2]].values
    targets = prototypes[:,2]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:3]
    synapses = perceptron(prototypes_biased, synapses, targets, 200, beta, lambda v: 0 if v < 0 else 1)
    x,y = get_separation_line(synapses)

    data_recall = read_csv(recall_data_file, header=None)
    prototypes_recall = data_recall[[0,1,2]].values
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased_recall = np.hstack((bias, prototypes))[:,:3]
    predict_targets = perceptron_outputs_act(prototypes_biased_recall, synapses, lambda v: 0 if v < 0 else 1)

    predicted_t = []
    for i in range(len(targets)):
        if targets[i] == predict_targets[i]:
            predicted_t.append(True)
        else:
            predicted_t.append(False)
    recall_figure_2D(prototypes_recall, predicted_t, axes)
    axes.axline(x, y, color='00')
    axes.set_autoscalex_on(False)
    axes.set_autoscaley_on(False)

def perceptron_train_3D(train_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0], projection='3d')
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    fig_ax2 = figure.add_subplot(gs[0, 1], projection='3d')
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 4)
    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,1,2,3]].values
    targets = prototypes[:,3]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:4]

    plot_prototypes_3D(prototypes, fig_ax1)

    old_synapses = []
    for i in range(epochs):
        x,y,z = get_separation_plane(synapses)
        classifications = perceptron_outputs_act(prototypes_biased, synapses, lambda v: 0 if v < 0 else 1)
        update_plane_figure(prototypes, fig_ax2, classifications, x, y, z)
        update_class_figure_3D(prototypes, classifications, fig_ax3)
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased, synapses, targets, 1, beta, lambda v: 0 if v < 0 else 1)
        plt.pause(0.7)

def perceptron_recall_3D(train_data_file, recall_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    axes = figure.add_subplot(gs[:,:], projection='3d')
    axes.legend(handles=[mpatches.Patch(color='green', label='Correct predictions'),
                         mpatches.Patch(color='red', label='Incorrect predictions')])
    figure.show()

    synapses = getRandomArray(-0.5, 0.5, 4)
    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,1,2,3]].values
    targets = prototypes[:,3]
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))[:,:4]
    synapses = perceptron(prototypes_biased, synapses, targets, 200, beta, lambda v: 0 if v < 0 else 1)
    x,y,z = get_separation_plane(synapses)

    data_recall = read_csv(recall_data_file, header=None)
    prototypes_recall = data_recall[[0,1,2,3]].values
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased_recall = np.hstack((bias, prototypes))[:,:4]
    predict_targets = perceptron_outputs_act(prototypes_biased_recall, synapses, lambda v: 0 if v < 0 else 1)

    predicted_t = []
    for i in range(len(targets)):
        if targets[i] == predict_targets[i]:
            predicted_t.append(True)
        else:
            predicted_t.append(False)
    recall_figure_3D(prototypes_recall, predicted_t, axes)

    points = [np.array(x), np.array(y), np.array(z)]
    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]
    
    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
    point  = np.array(p0)
    normal = np.array(u_cross_v)
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(2), range(2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    axes.plot_surface(xx, yy, z, alpha=0.2)

def perceptron_flowers_train(train_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1'),
                            mpatches.Patch(color='green', label='Class 2')])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()
    
    synapses = getRandomArray(-0.5, 0.5, 3)
    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,2,4]].values
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
    for i in range(epochs):
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

def perceptron_flowers_recall(train_data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    fig_ax1 = figure.add_subplot(gs[:,:])
    fig_ax1.legend(handles=[mpatches.Patch(color='green', label='Correct predictions'),
                            mpatches.Patch(color='red', label='Incorrect predictions')])
    figure.show()
    
    synapses = getRandomArray(-0.5, 0.5, 3)
    train_data = read_csv(train_data_file, header=None)
    prototypes = train_data[[0,2,4]].values
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

    synapses_setosa = perceptron(prototypes_biased, synapses, targets_setosa, epochs, beta, lambda v: 0 if v < 0 else 1)
    synapses_versicolor = perceptron(prototypes_biased, synapses, targets_versicolor, epochs, beta, lambda v: 0 if v < 0 else 2)
    synapses_virginica = perceptron(prototypes_biased, synapses, targets_virginica, epochs, beta, lambda v: 0 if v < 0 else 3)
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
