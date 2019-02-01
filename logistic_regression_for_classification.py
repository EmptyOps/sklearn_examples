#!/usr/bin/env python3

"""
Flower classification on the Iris dataset using a Naive Bayes
classifier and TensorFlow.
For more info: http://nicolovaligi.com/naive-bayes-tensorflow.html
"""

#from IPython import embed
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib import colors

from numpy import array
import sys, traceback
import json
import random

from sklearn import linear_model, datasets

checkpoint_directory = sys.argv[5]

#input_non_norm = json.load( open( sys.argv[1] ) )
#class_labels = json.load( open( sys.argv[2] ) )

#input_non_norm = array( input_non_norm )
#Y = array( class_labels )

#prediction_decision_id = sys.argv[6]

to_be_predicted_index = sys.argv[7] 

print( sys.argv[8] )

##input_new = tf.to_float(input_non_norm)
##X = tf.Session().run( input_new )
#X = input_non_norm
#print("X length")
#print(len(X))
#print("X[0] length")
#print(len(X[0]))

#input_to_be_predicted = json.load( open( sys.argv[3] ) )
#input_to_be_predicted = array( input_to_be_predicted )

#print("training logistic regression classification model for prediction_decision_id= " + prediction_decision_id)            

if __name__ == '__main__':

    tmpArr = json.load( open( sys.argv[2] ) )
    print(tmpArr)
    #print(tmpArr[0])
    prediction_decision_idArr = tmpArr[0].split("|")
    
    size = len(prediction_decision_idArr)
    for i in range(size):
    
        try:
            input_non_norm = json.load( open( sys.argv[1] + "\\1_" + prediction_decision_idArr[i] + ".json" ) )
            class_labels = json.load( open( sys.argv[1] + "\\2_" + prediction_decision_idArr[i] + ".json" ) )

            input_non_norm = array( input_non_norm )
            Y = array( class_labels )

            prediction_decision_id = prediction_decision_idArr[i]
            

            print( sys.argv[8] )

            ##input_new = tf.to_float(input_non_norm)
            ##X = tf.Session().run( input_new )
            X = input_non_norm
            print("X length")
            print(len(X))
            print("X[0] length")
            print(len(X[0]))

            input_to_be_predicted = json.load( open( sys.argv[1] + "\\3_" + prediction_decision_idArr[i] + ".json" ) )
            input_to_be_predicted = array( input_to_be_predicted )

            print("training logistic regression classification model at i="+str(i)+" for prediction_decision_id= " + prediction_decision_id)            


            # import some data to play with
            #iris = datasets.load_iris()
            #X = iris.data[:, :2]  # we only take the first two features.
            #Y = iris.target

            h = .02  # step size in the mesh

            logreg = linear_model.LogisticRegression(C=1e5)

            # we create an instance of Neighbours Classifier and fit the data.
            logreg.fit(X, Y)

            print(to_be_predicted_index)
            if to_be_predicted_index != '-1': 
                index = int(to_be_predicted_index)
                print(input_to_be_predicted[index])
                #res = gnb.predict([input_to_be_predicted[index]])
                res = logreg.predict([input_to_be_predicted[index]])
                #print(type(res))
                print(res)
                
                with open( checkpoint_directory + 'o_'+prediction_decision_id+'.json', 'w') as outfile:
                    json.dump(res.tolist(), outfile)


            """
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
            """
        except Exception as e: 
             #print('An error occured.')
             print(e)
             traceback.print_exc()
             with open( checkpoint_directory + 'o_'+prediction_decision_id+'.json', 'w') as outfile:
                    json.dump([8], outfile)
