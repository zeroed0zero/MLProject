from Src.common import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import read_csv
import numpy as np

def svm_train_2D(train_data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    train_data = read_csv(train_data_file, header=None)[[0,1,2]].values
    X = train_data[:,:2]
    y = train_data[:,2]

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X, y)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax1.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
    ax2.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
    
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax2.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])
    ax2.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')

    update_class_figure_2D(train_data, clf.predict(X), ax3)

def svm_recall_2D(train_data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    train_data = read_csv(train_data_file, header=None)[[0,1,2]].values
    X = train_data[:,:2]
    y = train_data[:,2]

    recall_data = read_csv(recall_data_file, header=None)[[0,1,2]].values
    X_t = recall_data[:,:2]
    y_t = recall_data[:,2]

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X, y)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax1.scatter(X_t[:,0], X_t[:,1], c=y, cmap=cm_bright, marker='.')
    ax2.scatter(X_t[:,0], X_t[:,1], c=y, cmap=cm_bright, marker='.')
    
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax2.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])
    ax2.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')

    update_class_figure_2D(recall_data, clf.predict(X_t), ax3)

def svm_train_flowers(train_data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1'),
                        mpatches.Patch(color='green', label='Class 2')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()
    
    train_data = read_csv(train_data_file, header=None)[[0,2,4]].values
    for i in range(len(train_data)):
        if train_data[i][2] == 'Iris-setosa': train_data[i][2] = 0
        elif train_data[i][2] == 'Iris-versicolor': train_data[i][2] = 1
        elif train_data[i][2] == 'Iris-virginica': train_data[i][2] = 2
    X = train_data[:,:2]
    y = train_data[:,2].astype('int')

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X, y)

    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#14681b'])
    ax1.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
    ax2.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, marker='.')
    
    X0,X1 = X[:,0], X[:,1]
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.4)

    update_class_figure_2D(train_data, clf.predict(X), ax3)

def svm_recall_flowers(train_data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0])
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1'),
                        mpatches.Patch(color='green', label='Class 2')])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()
    
    train_data = read_csv(train_data_file, header=None)[[0,2,4]].values
    for i in range(len(train_data)):
        if train_data[i][2] == 'Iris-setosa': train_data[i][2] = 0
        elif train_data[i][2] == 'Iris-versicolor': train_data[i][2] = 1
        elif train_data[i][2] == 'Iris-virginica': train_data[i][2] = 2
    X = train_data[:,:2]
    y = train_data[:,2].astype('int')

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)

    cm_bright = ListedColormap(['#FF0000', '#0000FF', '#14681b'])
    ax1.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, marker='.')
    ax2.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, marker='.')
    
    X0,X1 = X_train[:,0], X_train[:,1]
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.4)

    update_class_figure_2D(np.c_[X_test, y_test], clf.predict(X_test), ax3)
