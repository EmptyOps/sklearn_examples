#!/usr/bin/env python3

"""
Flower classification on the Iris dataset using a Naive Bayes
classifier and TensorFlow.
For more info: http://nicolovaligi.com/naive-bayes-tensorflow.html
"""

from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from numpy import array
import sys
import json
import random

from sklearn import linear_model, datasets


ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
input_non_norm = json.load( open( sys.argv[2] ) )
class_labels = json.load( open( sys.argv[8] ) )

input_non_norm = array( input_non_norm )
Y = array( class_labels )

to_be_predicted_index = -1 

X = input_non_norm
print("X length " + str(len(X)) )
print("X[0] length " + str(len(X[0])) )
print("Y length " + str(len(Y)) )
print(Y)

input_to_be_predicted = json.load( open( sys.argv[3] ) )
input_to_be_predicted = array( input_to_be_predicted )
input_to_be_predicted_labels = np.array( json.load( open( sys.argv[14] ) ) ) if len(sys.argv) >= 15 else None
outfile_path = sys.argv[15] if len(sys.argv) >= 16 else None

if __name__ == '__main__':

    # import some data to play with
    #iris = datasets.load_iris()
    #X = iris.data[:, :2]  # we only take the first two features.
    #Y = iris.target

    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    print(to_be_predicted_index)
    if to_be_predicted_index != -1: 
        index = int(to_be_predicted_index)
        
        print(input_to_be_predicted[index])
        #res = gnb.predict([input_to_be_predicted[index]])
        res = logreg.predict([input_to_be_predicted[index]])
        
        #print(type(res))
        print(res)
        
        with open( outfile_path, 'w') as outfile:
            json.dump(res.tolist(), outfile)
    else:
        print( "input " + str(len(input_to_be_predicted)) )
        print( input_to_be_predicted )
    
        res = logreg.predict(input_to_be_predicted)
        
        print( "result " + str(len(res)) )
        print( res )
        
        print("target")
        print( input_to_be_predicted_labels )

        if len(input_to_be_predicted_labels) > 0:
            from cross_entropy import cross_entropy
            print("cross_entropy")
            print( cross_entropy(res, input_to_be_predicted_labels) )
        
        with open( outfile_path, 'w') as outfile:
            json.dump(res.tolist(), outfile)
        
        # in dev mode 
        if False and ENV == 1:
            X_cls1_wrong = []
            X_cls2_wrong = []
            Y = input_to_be_predicted_labels
            
            Y_tmp = res
            unique, counts = np.unique(input_to_be_predicted_labels, return_counts=True)
            cnts = dict(zip(unique, counts))
            cnt1 = cnts[0]
            cnt2 = cnts[1]
            print( cnts )
            print( "cnt1 " + str(cnt1) + " cnt2 " + str(cnt2) )
            size1 = len(input_to_be_predicted_labels)
            
            big_class = 0 if cnt1 > cnt2 else 1
            for idx in range(0, size1):
                #TODO temp 
                #size2 = len(X[idx])
                #for idx2 in range(0, size2):
                #    X[idx][idx2] = X[idx][idx2] - 1000 if X[idx][idx2] > 1000 else X[idx][idx2]
            
                if Y_tmp[idx] == 0 and not Y_tmp[idx] == Y[idx] and len(X_cls1_wrong) < 10000:
                    X_cls1_wrong.append(X[idx])
                elif Y_tmp[idx] == 1 and not Y_tmp[idx] == Y[idx] and len(X_cls2_wrong) < 10000:
                    X_cls2_wrong.append(X[idx])
            
            print( "X_cls1_wrong " + str(len(X_cls1_wrong)) )
            if len(X_cls1_wrong) > 0:
                plt.subplot(2, 2, 1)
                plt.plot(X_cls1_wrong, color="yellow") 

                print( "X_cls2_wrong " + str(len(X_cls2_wrong)) )
                if len(X_cls2_wrong) > 0:
                    plt.subplot(2, 2, 2)
                    plt.plot(X_cls2_wrong, color="blue") 

            plt.show()

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if len(X[0]) == 1:
        # and plot the result
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.title('pd id = '+ prediction_decision_id)
        
        Z = logreg.predict(np.c_[X.ravel()])
        Z = Z.reshape(X.shape)
        plt.pcolormesh(X, Y, Z, cmap=plt.cm.Paired)
        
        plt.scatter(X.ravel(), Y, c=Y, edgecolors='k', cmap=plt.cm.Paired, zorder=20)

        plt.show()
    elif len(X[0]) == 2:
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()
