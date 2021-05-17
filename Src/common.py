import numpy as np
from matplotlib import pyplot as plt

class0_color_shape = 'r+'
class1_color_shape = 'b+'
class2_color_shape = 'g+'
class_colors_shapes = ['r+', 'b+', 'g+']
correct_color_shape = 'g+'
incorrect_color_shape = 'r+'

def get_number_of_classes(prototypes):
    return len(set(prototypes[:, len(prototypes[0]) - 1]))

def get_separation_line(synapses):
    return ((synapses[0]/synapses[1], 0), (0, synapses[0]/synapses[2]))

def update_line_figure(prototypes, classifications, figure, x, y):
    figure.cla()
    classified_0 = []
    classified_1 = []
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

def update_pred_figure(prototypes, figure, classifications):
    figure.cla()
    classes_x = [[] for i in range(get_number_of_classes(prototypes))]
    classes_y = [[] for i in range(get_number_of_classes(prototypes))]
    for i in range(get_number_of_classes(prototypes)):
        for j in range(len(prototypes)):
            classes_x[classifications[j]].append(prototypes[j, 0])
            classes_y[classifications[j]].append(prototypes[j, 1])
    
    for i in range(get_number_of_classes(prototypes)):
        figure.plot(classes_x[i], classes_y[i], class_colors_shapes[i])

def recall_figure_2D(prototypes, predicted_t, figure):
    classified_correct = []
    classified_incorrect = []
    for i in range(len(prototypes)):
        if predicted_t[i] == True:
            classified_correct.append([prototypes[i, 0], prototypes[i, 1]])
        else:
            classified_incorrect.append([prototypes[i, 0], prototypes[i, 1]])
    classified_correct = np.array(classified_correct)
    classified_incorrect = np.array(classified_incorrect)
    if len(classified_correct) != 0:
        figure.plot(classified_correct[:,0], classified_correct[:,1], correct_color_shape)
    if len(classified_incorrect) != 0:
        figure.plot(classified_incorrect[:,0], classified_incorrect[:,1], incorrect_color_shape)

# Plot for the classification each prototype received
def update_class_figure_2D(prototypes, classifications, figure):
    update_class_figure(prototypes, figure, classifications, 2)

def plot_prototypes_2D(prototypes, figure):
    proto_class = lambda p, c, col : p[p[:,2] == c][:,:2][:,col]
    for i in range(get_number_of_classes(prototypes)):
        figure.plot(proto_class(prototypes, i, 0), proto_class(prototypes, i, 1), class_colors_shapes[i])

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

def update_plane_figure(prototypes, figure, x, y, z):
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
    classifications = classify_prototypes_3D(prototypes, x, y, z)
    update_class_figure(prototypes, figure, classifications, 3)

def update_class_figure(prototypes, figure, classifications, dim):
    figure.cla()
    classes_x = [[] for i in range(get_number_of_classes(prototypes))]
    classes_y = [[] for i in range(get_number_of_classes(prototypes))]
    for i in range(get_number_of_classes(prototypes)):
        for j in range(len(prototypes)):
            if prototypes[j, dim] == i:
                classes_x[i].append(1/len(prototypes) * j)
                classes_y[i].append(classifications[j])

    for i in range(len(classes_x)):
        figure.plot(classes_x[i], classes_y[i], class_colors_shapes[i])

def plot_prototypes_3D(prototypes, figure):
    proto_class = lambda p, c, col : p[p[:,3] == c][:,:3][:,col]
    for i in range(get_number_of_classes(prototypes)):
        figure.plot(proto_class(prototypes, i, 0), proto_class(prototypes, i, 1), proto_class(prototypes, i, 2), class_colors_shapes[i])
    figure.set_autoscalex_on(False)
    figure.set_autoscaley_on(False)
