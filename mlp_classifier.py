from sklearn.neural_network import MLPClassifier
import sys
import json
from numpy import array
import numpy as np


ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
input_non_norm = json.load( open( sys.argv[2] ) )
class_labels = json.load( open( sys.argv[8] ) )

input_non_norm = array( input_non_norm )
Y = array( class_labels )

to_be_predicted_index = -1 

X = input_non_norm

input_to_be_predicted = np.array( json.load( open( sys.argv[3] ) ) )

input_to_be_predicted_labels = np.array( json.load( open( sys.argv[14] ) ) ) if len(sys.argv) >= 15 else None


if __name__ == '__main__':
    print("X length " + str(len(X)) )
    print("X[0] length " + str(len(X[0])) )
    print("Y length " + str(len(Y)) )
    print(Y)

    #X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    y = Y #[0, 0, 0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-1, activation='identity', verbose=False,
                        tol=1e-20, max_iter=10000, 
                        hidden_layer_sizes=(25, 25, 25, 25), random_state=1)
    print(clf.fit(X, y))   


    #print("weights between input and first hidden layer:")
    #print(clf.coefs_[0])
    #print("\nweights between first hidden and second hidden layer:")
    #print(clf.coefs_[1])


    print("w0 = ", clf.coefs_[0][0][0])
    #print("w1 = ", clf.coefs_[0][1][0])


    #for i in range(len(clf.coefs_)):
    #    number_neurons_in_layer = clf.coefs_[i].shape[1]
    #    for j in range(number_neurons_in_layer):
    #        weights = clf.coefs_[i][:,j]
    #        print(i, j, weights, end=", ")
    #        print()
    #    print()


    print("Bias values for first hidden layer:")
    print(clf.intercepts_[0])
    #print("\nBias values for second hidden layer:")
    #print(clf.intercepts_[1])

    
    print("target")
    print( input_to_be_predicted_labels )

    result = clf.predict( input_to_be_predicted )
    
    print("result")
    print(result)                      
                          
                          
    prob_results = clf.predict_proba( input_to_be_predicted )
    print(prob_results)
                      