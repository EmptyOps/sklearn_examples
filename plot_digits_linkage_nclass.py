"""
=============================================================================
Extended from Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the behavior "rich getting richer" of
agglomerative clustering that tends to create uneven cluster sizes.
This behavior is pronounced for the average linkage strategy,
that ends up with a couple of singleton clusters, while in the case
of single linkage we get a single central cluster with all other clusters
being drawn from noise points around the fringes.

=============================================================================
arguments
=============================================================================


=============================================================================
sections
=============================================================================
see section "unsupervised class and label detection" below

predict:
see section "predict class or labels" below

"""

# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

print(__doc__)
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets
import sys
import json


print(sys.argv)
ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
n_classes_ = int(sys.argv[4]) if len(sys.argv) >= 5 else 3
n_labels_ = int(sys.argv[5]) if len(sys.argv) >= 6 else 2
class_edge_colors = [ 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red', 'orange', 'blue', 'red' ]
n_components_ = int(sys.argv[6]) if len(sys.argv) >= 7 else 2
input_to_be_predicted = None


#
#   unsupervised class and label detection
#

if ENV == 0:
    digits = datasets.load_digits(n_class=10)
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
else:
    X = np.array( json.load( open( sys.argv[2] ) ) )
    y = np.zeros( len(X) )
    
    input_to_be_predicted = np.array( json.load( open( sys.argv[3] ) ) )


np.random.seed(0)


#TODO temp. randomize input_to_be_predicted and assumed no class for y
if not ENV == 0:
    from random import randint
    for i, val in enumerate(input_to_be_predicted):
        randi = randint(0, len(X)-1)
        input_to_be_predicted[i] = X[randi]
        X[randi] = val


print("X=" + str(len(X)) + " Y=" + str(len(y)))

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


if ENV == 0:
    X, y = nudge_images(X, y)
    print("X=" + str(len(X)) + " Y=" + str(len(y)))


#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=n_components_).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'): #, 'single'
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_classes_)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
    y = clustering.labels_
    
    #print("prediction with " + linkage)
    #input_to_be_predicted_red = manifold.SpectralEmbedding(n_components=n_components_).fit_transform(input_to_be_predicted)
    #print( clustering.fit_predict(input_to_be_predicted_red[0]) )
    
if ENV == 0:    
    plt.show()


#########################################################################################################################################################
#########                END first part of the program 
#########################################################################################################################################################


#    
#   predict class or labels    
#   for reference refer plot_multilabel.py example    
#
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

n_components_ = int(sys.argv[7]) if len(sys.argv) >= 8 else 2
is_use_transform = False

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

def plot_subfigure(X, Y, subplot, title, transform, input_to_be_predicted):

    #print("X=")
    #print(X)
    #print("Y=")
    #print(Y)
    if transform == "pca":
        if is_use_transform == True:
            pcaobj = PCA(n_components=n_components_)
            X = pcaobj.fit_transform(X)
            #input_to_be_predicted = pcaobj.fit_transform(input_to_be_predicted)
    elif transform == "cca":
        if is_use_transform == True:    
            ccaobj = CCA(n_components=n_components_).fit(X, Y)
            X = ccaobj.transform(X)
            #input_to_be_predicted = ccaobj.transform(input_to_be_predicted)
    else:
        raise ValueError

        
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    
    print("X=")
    print(X)
    print("Y=")
    print(Y)
    print("X length " + str(len(X)))
    print("Y length " + str(len(Y)))
    print("n_components_=" + str(n_components_))

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    if not ENV == 0:
        print("predict")
        print( len(input_to_be_predicted) )
        print( classif.predict(input_to_be_predicted) )
    
    plt.subplot(2, 2, subplot)
    plt.title(title)

    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))

    for idx in np.arange(n_classes_):
        #zero_class = np.where(Y[:, 0])
        #one_class = np.where(Y[:, 1])
        #plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b', facecolors='none', linewidths=2, label='Class 1')
        #plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange', facecolors='none', linewidths=2, label='Class 2')
        plt.scatter(X[np.where(Y[:, idx]), 0], X[np.where(Y[:, idx]), 1], s=160, edgecolors=class_edge_colors[idx], facecolors='none', linewidths=2, label='Class ' + str(idx))

        #plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--', 'Boundary\nfor class 1')
        #plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.', 'Boundary\nfor class 2')
        plot_hyperplane(classif.estimators_[idx], min_x, max_x, 'k' + ( '-.' if idx == 2 else ( '-o' if idx == 1 else '-+' ) ), 'Boundary\nfor class '+str(idx))
                        
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")

#tranform y        
if ENV == 0:
    #print( type(Y) )
    #print( Y[0] )
    tmp = ''
else:
    from sklearn import preprocessing
    #y_tranformed = []
    #for i, val in enumerate(y):
    #    print( "class labels..." )
    #    print(val)
    #    y_tranformed.append( [i, val] )
    #y_tranformed = np.array( y_tranformed )    
    lb = preprocessing.LabelBinarizer(sparse_output=False)
    print( lb.fit(y) )
    y_tranformed = lb.transform(y)    #lb.classes_
    print("y_tranformed result")    
    print(y_tranformed)    
    

if ENV == 0:
    plt.figure(figsize=(8, 6))

    X, Y = make_multilabel_classification(n_samples=1000, n_features=540, n_classes=n_classes_, n_labels=n_labels_,
                                          allow_unlabeled=True,
                                          random_state=1)
else:
    #X = X_red
    Y = y_tranformed

if ENV == 0:
    print( type(Y) )
    print( Y[0] )
    
plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca", input_to_be_predicted)
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca", input_to_be_predicted)
    

if ENV == 0:
    X, Y = make_multilabel_classification(n_samples=1000, n_features=540, n_classes=n_classes_, n_labels=n_labels_,
                                          allow_unlabeled=False,
                                          random_state=1)
else:
    #X = X_red
    Y = y_tranformed

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca", input_to_be_predicted)
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca", input_to_be_predicted)


if True or ENV == 0:
    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
    plt.show()

