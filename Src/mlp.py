from Data.random_models import *
from Src.common import *
from matplotlib.colors import ListedColormap
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def mlp_train_2D(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                            mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, 0])
    ax4 = figure.add_subplot(gs[1, 1])
    figure.show()

    data = read_csv(data_file, header=None)[[0,1,2]].values
    X = data[:,:2]
    y = data[:,2]

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax1.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')

    clf = MLPClassifier(random_state=1, max_iter=epochs, hidden_layer_sizes=(256,256,256), activation='tanh', solver='adam', learning_rate_init=beta)
    clf.partial_fit(X, y, [0,1])

    epoch_values = []
    mse_values = []
    for i in range(epochs):
        classifications = clf.predict(X)
        epoch_values.append(i)
        mse_values.append(mean_squared_error(y, classifications))

        clf.partial_fit(X, y)
        ax2.cla()
        ax2.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
        ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.3)

        update_class_figure_2D(data, classifications, ax3)
        update_mse_figure(mse_values, epoch_values, ax4)
        plt.pause(0.00001)

def mlp_recall_2D(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    ax1 = figure.add_subplot(gs[:, :])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1')])
    figure.show()

    data = read_csv(data_file, header=None)[[0,1,2]].values
    X = data[:,:2]
    y = data[:,2].astype('int')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    clf = MLPClassifier(random_state=1, max_iter=epochs, hidden_layer_sizes=(256,256,256), activation='tanh', solver='adam', learning_rate_init=beta)
    clf.fit(X_train, y_train)
    
    y_test_predict = clf.predict(X_test)
    ax1.plot(list(range(len(y_test))), y_test, 'b.')
    ax1.plot(list(range(len(y_test_predict))), y_test_predict, 'ro', fillstyle='none')

def mlp_train_3D(data_file, beta, epochs):
    print("Not implemented yet\n")
def mlp_recall_3D(data_file, beta, epochs):
    print("Not implemented yet\n")

def mlp_train_flowers(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1'),
                        mpatches.Patch(color='green', label='Class 2')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, 0])
    ax4 = figure.add_subplot(gs[1, 1])
    figure.show()
    
    data = read_csv(data_file, header=None)[[0,2,4]].values
    for i in range(len(data)):
        if data[i][2] == 'Iris-setosa': data[i][2] = 0
        elif data[i][2] == 'Iris-versicolor': data[i][2] = 1
        elif data[i][2] == 'Iris-virginica': data[i][2] = 2
    X = data[:,:2]
    y = data[:,2].astype('int')

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))

    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
    ax1.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')

    clf = MLPClassifier(random_state=1, max_iter=epochs, hidden_layer_sizes=(256,256,256), activation='tanh', solver='adam', learning_rate_init=beta)
    clf.partial_fit(X, y, [0,1,2])

    epoch_values = []
    mse_values = []
    for i in range(epochs):
        clf.partial_fit(X, y)
        classifications = clf.predict(X)
        epoch_values.append(i)
        mse_values.append(mean_squared_error(y, classifications))

    ax2.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.3)
    update_class_figure_2D(data, classifications, ax3)
    update_mse_figure(mse_values, epoch_values, ax4)

def mlp_recall_flowers(data_file, beta, epochs):
    figure = plt.figure()
    gs = figure.add_gridspec(1, 1)
    ax = figure.add_subplot(gs[:, :])
    figure.show()
    
    data = read_csv(data_file, header=None)[[0,2,4]].values
    for i in range(len(data)):
        if data[i][2] == 'Iris-setosa': data[i][2] = 0
        elif data[i][2] == 'Iris-versicolor': data[i][2] = 1
        elif data[i][2] == 'Iris-virginica': data[i][2] = 2
    X = data[:,:2]
    y = data[:,2].astype('int')

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    clf = MLPClassifier(random_state=1, max_iter=epochs, hidden_layer_sizes=(256,256,256), activation='tanh', solver='adam', learning_rate_init=beta)
    clf.fit(X_train, y_train)

    y_test_predict = clf.predict(X_test)
    ax.plot(list(range(len(y_test))), y_test, 'b.')
    ax.plot(list(range(len(y_test_predict))), y_test_predict, 'ro', fillstyle='none')
