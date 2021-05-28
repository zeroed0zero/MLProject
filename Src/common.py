from matplotlib import pyplot as plt
import numpy as np

class0_color_shape = 'r.'
class1_color_shape = 'b.'
class2_color_shape = 'g.'
class_colors_shapes = ['r.', 'b.', 'g.']
correct_color_shape = 'g.'
incorrect_color_shape = 'r.'

def get_number_of_classes(X):
    return len(set(X[:, len(X[0]) - 1]))

def get_separation_line(synapses):
    return ((synapses[0]/synapses[1], 0), (0, synapses[0]/synapses[2]))

def update_line_figure(X, classifications, figure, x, y):
    figure.cla()
    classified_0 = []
    classified_1 = []
    for i in range(len(X)):
        if classifications[i] == 0:
            classified_0.append([X[i, 0], X[i, 1]])
        else:
            classified_1.append([X[i, 0], X[i, 1]])
    classified_0 = np.array(classified_0)
    classified_1 = np.array(classified_1)
    
    figure.axline(x, y, color='00')
    if len(classified_0) != 0:
        figure.plot(classified_0[:,0], classified_0[:,1], class0_color_shape)
    if len(classified_1) != 0:
        figure.plot(classified_1[:,0], classified_1[:,1], class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def update_line_figure_adaline(X, classifications, figure, x, y):
    figure.cla()
    classified_0 = []
    classified_1 = []
    for i in range(len(X)):
        if classifications[i] < 0:
            classified_0.append([X[i, 0], X[i, 1]])
        else:
            classified_1.append([X[i, 0], X[i, 1]])
    classified_0 = np.array(classified_0)
    classified_1 = np.array(classified_1)
    
    figure.axline(x, y, color='00')
    if len(classified_0) != 0:
        figure.plot(classified_0[:,0], classified_0[:,1], class0_color_shape)
    if len(classified_1) != 0:
        figure.plot(classified_1[:,0], classified_1[:,1], class1_color_shape)
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def update_mse_figure(mse, epoch, figure):
    figure.cla()
    figure.plot(epoch, mse, 'k-')

def update_pred_figure(X, figure, classifications):
    figure.cla()
    classes_x = [[] for i in range(get_number_of_classes(X))]
    classes_y = [[] for i in range(get_number_of_classes(X))]
    for i in range(get_number_of_classes(X)):
        for j in range(len(X)):
            classes_x[classifications[j]].append(X[j, 0])
            classes_y[classifications[j]].append(X[j, 1])
    
    for i in range(get_number_of_classes(X)):
        figure.plot(classes_x[i], classes_y[i], class_colors_shapes[i])

def recall_figure_2D(X, predicted_t, figure):
    classified_correct = []
    classified_incorrect = []
    for i in range(len(X)):
        if predicted_t[i] == True:
            classified_correct.append([X[i, 0], X[i, 1]])
        else:
            classified_incorrect.append([X[i, 0], X[i, 1]])
    classified_correct = np.array(classified_correct)
    classified_incorrect = np.array(classified_incorrect)
    if len(classified_correct) != 0:
        figure.plot(classified_correct[:,0], classified_correct[:,1], correct_color_shape)
    if len(classified_incorrect) != 0:
        figure.plot(classified_incorrect[:,0], classified_incorrect[:,1], incorrect_color_shape)

# Plot for the classification each prototype received
def update_class_figure_2D(X, classifications, figure):
    update_class_figure(X, figure, classifications, 2)

def plot_prototypes_2D(X, figure):
    proto_class = lambda p, c, col : p[p[:,2] == c][:,:2][:,col]
    for i in range(get_number_of_classes(X)):
        figure.plot(proto_class(X, i, 0), proto_class(X, i, 1), class_colors_shapes[i])

def update_class_figure(X, figure, classifications, dim):
    figure.cla()
    classes_x = []
    classes_y = []
    for i in range(get_number_of_classes(X)):
        classes_x.append([])
        classes_y.append([])
        for j in range(len(X)):
            if X[j, dim] == i:
                classes_x[i].append(1/len(X) * j)
                classes_y[i].append(classifications[j])
    for i in range(len(classes_x)):
        figure.plot(classes_x[i], classes_y[i], class_colors_shapes[i])

# Plot for the classification each prototype received
def update_class_figure_3D(X, classifications, figure):
    update_class_figure(X, figure, classifications, 3)

def plot_prototypes_3D(X, figure):
    proto_class = lambda p, c, col : p[p[:,3] == c][:,:3][:,col]
    for i in range(get_number_of_classes(X)):
        figure.plot(proto_class(X, i, 0), proto_class(X, i, 1), proto_class(X, i, 2), class_colors_shapes[i])
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)

def get_separation_plane(synapses):
    return ((synapses[0]/synapses[1], 0, 0), (0, synapses[0]/synapses[2], 0), (0, 0, synapses[0]/synapses[3]))

def update_plane_figure(X, figure, classifications, x, y, z):
    figure.cla()
    classified_0 = []
    classified_1 = []
    for i in range(len(X)):
        if classifications[i] == 0:
            classified_0.append([X[i, 0], X[i, 1], X[i, 2]])
        else:
            classified_1.append([X[i, 0], X[i, 1], X[i, 2]])
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

def recall_figure_3D(X, predicted_t, figure):
    classified_correct = []
    classified_incorrect = []
    for i in range(len(X)):
        if predicted_t[i] == True:
            classified_correct.append([X[i, 0], X[i, 1], X[i, 2]])
        else:
            classified_incorrect.append([X[i, 0], X[i, 1], X[i, 2]])
    classified_correct = np.array(classified_correct)
    classified_incorrect = np.array(classified_incorrect)
    if len(classified_correct) != 0:
        figure.plot(classified_correct[:,0], classified_correct[:,1], classified_correct[:,2], correct_color_shape)
    if len(classified_incorrect) != 0:
        figure.plot(classified_incorrect[:,0], classified_incorrect[:,1], classified_incorrect[:,2], incorrect_color_shape)

def mse(X, synapses, targets):
    return (1/len(X)) * sum([(targets[i] - np.matmul(synapses, X[i]))**2 for i in range(len(X))])
