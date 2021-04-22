import numpy as np

class0_color_shape = 'r+'
class1_color_shape = 'b+'

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
    return ((synapses[0]/synapses[1], 0), (0, synapses[0]/synapses[2]))

# If the prototype is above the line it is classified as class 1, otherwise as class 0
def classify_prototypes_2D(prototypes, x, y):
    classifications = []
    for i in range(len(prototypes)):
        v1 = (-x[0], y[1])
        v2 = (-prototypes[i, 0], y[1] - prototypes[i,1])
        cp = v1[0]*v2[1] - v1[1]*v2[0] # Cross product
        if cp < 0:
            classifications.append(0)
        else:
            classifications.append(1)
    return classifications

def update_line_figure_2D(prototypes, figure, x, y):
    figure.cla()
    classified_0 = []
    classified_1 = []
    classifications = classify_prototypes_2D(prototypes, x, y)
    for i in range(len(prototypes)):
        if classifications[i] == 0:
            classified_0.append([prototypes[i, 0], prototypes[i, 1]])
        else:
            classified_1.append([prototypes[i, 0], prototypes[i, 1]])
    classified_0 = np.array(classified_0)
    classified_1 = np.array(classified_1)
    
    figure.axline(x, y, color='00')
    if len(classified_0) != 0:
        figure.plot(classified_0[:,0], classified_0[:,1], class0_color_shape)
    if len(classified_1) != 0:
        figure.plot(classified_1[:,0], classified_1[:,1], class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

# Plot for the classification each prototype received
def update_class_figure_2D(prototypes, figure, x, y):
    figure.cla()
    classifications = classify_prototypes_2D(prototypes, x, y)

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
        figure.plot(x_c0, y_c0, class0_color_shape)
    if len(x_c1) != 0:
        figure.plot(x_c1, y_c1, class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def plot_prototypes_2D(prototypes, figure):
    c0 = prototypes[prototypes[:,2] == 'c0'][:,:2]
    c1 = prototypes[prototypes[:,2] == 'c1'][:,:2]

    figure.plot(c0[:,0], c0[:,1], class0_color_shape)
    figure.plot(c1[:,0], c1[:,1], class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def get_separation_plane(synapses):
    return ((synapses[0]/synapses[1], 0, 0), (0, synapses[0]/synapses[2], 0), (0, 0, synapses[0]/synapses[3]))

# If the prototype is above the plane it is classified as class 1, otherwise as class 0
def classify_prototypes_3D(prototypes, x, y, z):
    classifications = []
    for i in range(len(prototypes)):
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
        cp = prototypes[i, 0] * normal[0] + prototypes[i, 1] * normal[1] + prototypes[i, 2] * normal[2] + d
        if cp < 0: classifications.append(0)
        else: classifications.append(1)
    return classifications

def update_line_figure_3D(prototypes, figure, x, y, z):
    figure.cla()
    classified_0 = []
    classified_1 = []
    classifications = classify_prototypes_3D(prototypes, x, y, z)
    for i in range(len(prototypes)):
        if classifications[i] == 0:
            classified_0.append([prototypes[i, 0], prototypes[i, 1], prototypes[i, 2]])
        else:
            classified_1.append([prototypes[i, 0], prototypes[i, 1], prototypes[i, 2]])
    classified_0 = np.array(classified_0)
    classified_1 = np.array(classified_1)

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

    figure.plot_surface(xx, yy, z, alpha=0.2)
    if len(classified_0) != 0:
        figure.plot(classified_0[:,0], classified_0[:,1], classified_0[:,2], class0_color_shape)
    if len(classified_1) != 0:
        figure.plot(classified_1[:,0], classified_1[:,1], classified_1[:,2], class1_color_shape)
    figure.set_xlim3d(0, 1)
    figure.set_ylim3d(0, 1)
    figure.set_zlim3d(0, 1)

# Plot for the classification each prototype received
def update_class_figure_3D(prototypes, figure, x, y, z):
    figure.cla()
    classifications = classify_prototypes_3D(prototypes, x, y, z)

    x_c0 = []
    x_c1 = []
    y_c0 = []
    y_c1 = []
    for i in range(len(prototypes)):
        if prototypes[i, 3] == 'c0':
            x_c0.append(1/len(prototypes) * i)
            if classifications[i] == 0: y_c0.append(0)
            else: y_c0.append(1)
        elif prototypes[i, 3] == 'c1':
            x_c1.append(1/len(prototypes) * i)
            if classifications[i] == 1: y_c1.append(1)
            else: y_c1.append(0)

    if len(x_c0) != 0:
        figure.plot(x_c0, y_c0, class0_color_shape)
    if len(x_c1) != 0:
        figure.plot(x_c1, y_c1, class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def plot_prototypes_3D(prototypes, figure):
    c0 = prototypes[prototypes[:,3] == 'c0'][:,:3]
    c1 = prototypes[prototypes[:,3] == 'c1'][:,:3]

    figure.plot(c0[:,0], c0[:,1], c0[:,2], class0_color_shape)
    figure.plot(c1[:,0], c1[:,1], c1[:,2], class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def g():
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import numpy as np

    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0], projection='3d')
    fig_ax2 = figure.add_subplot(gs[0, 1], projection='3d')
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    data = read_csv('../Data/lin_sep_3d.csv', header=None)
    prototypes = data[[0,1,2,3]].values
    synapses = np.array([.1,.01,.01,.01])

    labels = data[3].values
    map_dict = {'c0':0, 'c1':1}
    targets = np.array([map_dict[label] for label in labels])

    beta = 0.0001
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))

    plot_prototypes_3D(prototypes, fig_ax1)

    x,y,z = get_separation_plane(synapses)
    update_line_figure_3D(prototypes, fig_ax2, x, y, z)
    update_class_figure_3D(prototypes, fig_ax3, x, y, z)

    old_synapses = []
    for i in range(1000):
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased[:,:4], synapses, targets, 1, beta, activation_func_0_1)

        print('x: {} y: {} z: {}'.format(x,y,z))
        x,y,z = get_separation_plane(synapses)
        update_line_figure_3D(prototypes, fig_ax2, x, y, z)
        update_class_figure_3D(prototypes, fig_ax3, x, y, z)
        plt.pause(0.7)
    plt.pause(200_000)

def f():
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import numpy as np

    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    fig_ax1 = figure.add_subplot(gs[0, 0])
    fig_ax2 = figure.add_subplot(gs[0, 1])
    fig_ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    data = read_csv('../Data/xor.csv', header=None)
    prototypes = data[[0,1,2]].values
    synapses = np.array([.4,.2,.2])

    labels = data[2].values
    map_dict = {'c0':0, 'c1':1}
    targets = np.array([map_dict[label] for label in labels])

    beta = 0.0001
    rows, _ = prototypes.shape
    bias = np.ones((rows, 1)) * -1
    prototypes_biased = np.hstack((bias, prototypes))

    plot_prototypes_2D(prototypes, fig_ax1)

    x,y = get_separation_line(synapses)
    update_line_figure_2D(prototypes, fig_ax2, x, y)
    update_class_figure_2D(prototypes, fig_ax3, x, y)

    old_synapses = []
    for i in range(1000):
        if np.array_equal(old_synapses, synapses): break
        old_synapses = synapses
        synapses = perceptron(prototypes_biased[:,:3], synapses, targets, 1, beta, activation_func_0_1)

        x,y = get_separation_line(synapses)
        update_line_figure_2D(prototypes, fig_ax2, x, y)
        update_class_figure_2D(prototypes, fig_ax3, x, y)
        plt.pause(0.7)
    plt.pause(200_000)

if __name__ == '__main__':
    g()
