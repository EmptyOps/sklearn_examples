#!/usr/bin/env python3

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

#do_classification default 'kmeans' 
do_classification = sys.argv[10] if len(sys.argv) >= 11 else 'kmeans' 

#do_regresssion default 'logistic' 
do_regresssion = sys.argv[11] if len(sys.argv) >= 12 else 'logistic' 

#mode default line 
mode = sys.argv[9] if len(sys.argv) >= 10 else 'line' 


#do_classification_adjust default 'if_y_available' 
do_classification_adjust = sys.argv[12] if len(sys.argv) >= 13 else 'if_y_available' 


if __name__ == '__main__':
    print("X length " + str(len(X)) )
    print("X[0] length " + str(len(X[0])) )
    print("Y length " + str(len(Y)) )
    print(Y)
    
    print( "do_classification " + do_classification )
    print( "mode " + mode )
    print( "do_classification_adjust " + do_classification_adjust )

    X_cls1 = []
    X_cls2 = []
    X_cls1_wrong = []
    X_cls2_wrong = []
    if not do_classification == '':
        if do_classification == 'kmeans':
            size1 = len(Y)

            #TODO temp 
            for idx in range(0, size1):
                size2 = len(X[idx])
                for idx2 in range(0, size2):
                    X[idx][idx2] = X[idx][idx2] + 1000 if X[idx][idx2] >= 0.5 else X[idx][idx2]            
                    X[idx][idx2] = - 1000 - X[idx][idx2] if X[idx][idx2] <= 0.05 else X[idx][idx2]            

            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import scale

            X = PCA(n_components=12).fit_transform(X)
            kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
            kmeans.fit(X)
            
            #adjust 
            if do_classification_adjust == 'if_y_available' and len(Y) > 0:
                Y_tmp = kmeans.labels_
                unique, counts = np.unique(Y_tmp, return_counts=True)
                cnts = dict(zip(unique, counts))
                cnt1 = cnts[0]
                cnt2 = cnts[1]
                print( cnts )
                print( "cnt1 " + str(cnt1) + " cnt2 " + str(cnt2) )
                
                big_class = 0 if cnt1 > cnt2 else 1
                for idx in range(0, size1):
                    #TODO temp 
                    #size2 = len(X[idx])
                    #for idx2 in range(0, size2):
                    #    X[idx][idx2] = X[idx][idx2] - 1000 if X[idx][idx2] > 1000 else X[idx][idx2]
                
                    if Y_tmp[idx] == big_class and not Y[idx] == 0 and len(X_cls1_wrong) < 10000:
                        X_cls1_wrong.append(X[idx])
                    elif not Y_tmp[idx] == big_class and not Y[idx] == 1 and len(X_cls2_wrong) < 10000:
                        X_cls2_wrong.append(X[idx])
                
                    if not Y_tmp[idx] == big_class and not Y[idx] == 1:
                        Y_tmp[idx] = big_class
                    
                    if Y_tmp[idx] == big_class:
                        X_cls1.append( X[idx] )
                    else:
                        X_cls2.append( X[idx] )
                    
                Y = Y_tmp        
                    
            else: 
                Y = kmeans.labels_

    if mode == 'line':
        print( "X_cls1_wrong " + str(len(X_cls1_wrong)) )
        if len(X_cls1_wrong) > 0:
            plt.subplot(2, 2, 1)
            plt.plot(X_cls1_wrong, color="green") 

            print( "X_cls2_wrong " + str(len(X_cls2_wrong)) )
            if len(X_cls2_wrong) > 0:
                plt.subplot(2, 2, 2)
                plt.plot(X_cls2_wrong, color="red") 
        elif len(X_cls1) > 0:
            plt.subplot(2, 2, 1)
            plt.plot(X_cls1, color="green") 

            plt.subplot(2, 2, 2)
            plt.plot(X_cls2, color="red") 
        else:
            plt.subplot(2, 2, 1)
            sizea = len(X)
            for idx in range(0, sizea):
                if( Y[idx] == 1 ):
                    plt.plot(X[idx], color="green") 

            plt.subplot(2, 2, 2)
            for idx in range(0, sizea):
                if not Y[idx] == 1:
                    plt.plot(X[idx], color="red") 
            
            
    plt.show()
    