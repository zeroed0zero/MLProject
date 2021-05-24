from Src.common import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
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

def svm_train_3D(train_data_file):
    figure = plt.figure()
    gs = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gs[0, 0], projection='3d')
    ax1.legend(handles=[mpatches.Patch(color='red', label='Class 0'),
                        mpatches.Patch(color='blue', label='Class 1')])
    ax2 = figure.add_subplot(gs[0, 1], projection='3d')
    ax3 = figure.add_subplot(gs[1, :])
    figure.show()

    train_data = read_csv(train_data_file, header=None)[[0,1,2,3]].values
    X = train_data[:,:3]
    y = train_data[:,3]

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X, y)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax1.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=cm_bright, marker='.')
    ax2.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=cm_bright, marker='.')
    
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    zlim = ax2.get_zlim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    zz = np.linspace(zlim[0], zlim[1], 30)
    ZZ, YY, XX = np.meshgrid(zz, yy, xx)
    xyz = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    Z = clf.decision_function(xyz).reshape(XX.shape)
    ax2.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])
    ax2.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], clf.support_vectors_[:,2], s=100, linewidth=1, facecolors='none')

    update_class_figure_3D(train_data, clf.predict(X), ax3)
