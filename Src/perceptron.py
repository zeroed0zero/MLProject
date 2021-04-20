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

def activation_func_0_1(v):
    if v < 0:
        return 0
    elif v >= 0:
        return 1

def get_separation_line(synapses):
    return (synapses[0]/synapses[1], synapses[0]/synapses[2])

def classify_prototypes(prototypes):
    classifications = []
    for i in range(len(prototypes)):
        v1 = (-x, y)
        v2 = (-prototypes[i, 0], y - prototypes[i,1])
        cp = v1[0]*v2[1] - v1[1]*v2[0] # Cross product
        if cp < 0:
            classifications.append(0)
        else:
            classifications.append(1)
    return classifications

def update_line_figure(prototypes, figure, x, y):
    figure.cla()
    classified_0 = []
    classified_1 = []
    classifications = classify_prototypes(prototypes)
    for i in range(len(prototypes)):
        if classifications[i] == 0:
            classified_0.append([prototypes[i, 0], prototypes[i, 1]])
        else:
            classified_1.append([prototypes[i, 0], prototypes[i, 1]])
    classified_0 = np.array(classified_0)
    classified_1 = np.array(classified_1)
    
    figure.axline((x,0), (0,y), color='00')
    if len(classified_0) != 0:
        figure.plot(classified_0[:,0], classified_0[:,1], 'r+')
    if len(classified_1) != 0:
        figure.plot(classified_1[:,0], classified_1[:,1], 'b+')
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def update_class_figure(prototypes, figure):
    figure.cla()
    classifications = classify_prototypes(prototypes)

    added_featur
    x_c0 = []
    x_c1 = []
    y_c0 = []
    y_c1 = []
    for i in range(len(prototypes)):
        if prototypes[i, 2] == 'c0':
            x_c0.append(1/len(prototypes) * i)
            if classifications[i] == 0: y_c0.append(0)
            else: y_c0.append(1)
        elif prototypes[i, 2] == 'c1':
            x_c1.append(1/len(prototypes) * i)
            if classifications[i] == 1: y_c1.append(1)
            else: y_c1.append(0)

    if len(x_c0) != 0:
        figure.plot(x_c0, y_c0, 'ro')
    if len(x_c1) != 0:
        figure.plot(x_c1, y_c1, 'bo')
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import numpy as np

    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    data = read_csv('../Data/lin_sep.csv', header=None)
    prototypes = data[[0,1,2]].values
    synapses = np.array([.4,.2,.2])

    labels = data[2].values
    map_dict = {'c0':0, 'c1':1}
    targets = np.array([map_dict[label] for label in labels])

    beta = 0.0001
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))
    c0 = prototypes[prototypes[:,2] == 'c0'][:,:2]
    c1 = prototypes[prototypes[:,2] == 'c1'][:,:2]

    fig_ax1.plot(c0[:,0], c0[:,1], 'r+')
    fig_ax1.plot(c1[:,0], c1[:,1], 'b+')
    fig_ax1.set_autoscalex_on(False)
    fig_ax1.set_autoscaley_on(False)

    x,y = get_separation_line(synapses)
    fig_ax2.axline((x,0), (0,y), color='00')

    update_line_figure(prototypes, fig_ax2, x, y)
    update_class_figure(prototypes, fig_ax3)

    old_synapses = []
    for i in range(1000):
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased[:,:3], synapses, targets, 1, beta, activation_func_0_1)

        x,y = get_separation_line(synapses)
        update_line_figure(prototypes, fig_ax2, x, y)
        update_class_figure(prototypes, fig_ax3)
        plt.pause(0.7)
    plt.pause(200_000)

