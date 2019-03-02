'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras import backend as K
from keras.models import load_model

# Keras
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

# Others
import string
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE


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

n_classes_ = int(sys.argv[4]) if len(sys.argv) >= 5 else 3

input_to_be_predicted = np.array( json.load( open( sys.argv[3] ) ) )
input_to_be_predicted_labels = np.array( json.load( open( sys.argv[14] ) ) ) if len(sys.argv) >= 15 else None

outfile_path = sys.argv[15] if len(sys.argv) >= 16 else None
outfile_path_prob = sys.argv[19] if len(sys.argv) >= 20 else None

rows = int(sys.argv[16]) if len(sys.argv) >= 17 else 0
cols = int(sys.argv[17]) if len(sys.argv) >= 18 else 0

modelfile_path = sys.argv[18] if len(sys.argv) >= 19 else None

is_sample_debug_only = int(sys.argv[20]) if len(sys.argv) >= 21 else 0


model = None

import os.path
if os.path.isfile(modelfile_path): 
    model = load_model(modelfile_path)    

    
# input image dimensions
img_rows, img_cols = rows, cols   #28, 28

x_test = input_to_be_predicted
    
if True or is_sample_debug_only == 0:    
    if model == None:    
        print("X length " + str(len(X)) )
        print("X[0] length " + str(len(X[0])) )
        print("Y length " + str(len(Y)) )
        print("Y length 0 class " + str(Y.count(0)) )
        print("Y length 1 class " + str(Y.count(1)) )
        print(Y)


        batch_size = 128
        num_classes = n_classes_ #10
        epochs = 100    #1000     #12

        # the data, split between train and test sets
        #(x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = X
        y_train = Y
        y_test = input_to_be_predicted_labels

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        #TODO turned off temporarily
        #x_train /= 255
        #x_test /= 255
        
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        #y_train = keras.utils.to_categorical(y_train, num_classes)
        #y_test = keras.utils.to_categorical(y_test, num_classes)

        
        ## Network architecture
        model = Sequential()
        model.add(Embedding(20000, 100, input_length=len(X[0]) ) )
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        ## Fit the model
        model.fit(x_train, y_train, validation_split=0.4, epochs=epochs)

        
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save( modelfile_path )
        
    else:
            
        x_test = x_test.astype('float32')
        
        #TODO
        #x_test /= 255


    #input_to_be_predicted = x_test    

    print("target")
    print( input_to_be_predicted_labels )

    prob_results = model.predict( x_test )
    res = prob_results.argmax(axis=-1)
    print("result")
    print(res)                      
    print(prob_results)
                      
    if not outfile_path == None:
        with open( outfile_path, 'w') as outfile:
            json.dump(res.tolist(), outfile)                      

    if not outfile_path_prob == None:
        with open( outfile_path_prob, 'w') as outfile:
            json.dump(prob_results.tolist(), outfile)                      
        
    
# in dev mode 
if ENV == 1:
    from matplotlib import pyplot as plt
    
    #check sample accuracy 
    if False:
        while(True):
            """
            columns = 10
            rows = 4
            fig, ax_array = plt.subplots(rows, columns,squeeze=False)
            for i,ax_row in enumerate(ax_array):
                for j,axes in enumerate(ax_row):
                    axes.set_title('{},{}'.format(i,j))
                    axes.set_yticklabels([])
                    axes.set_xticklabels([])
                    
            #         axes.plot(you_data_goes_here,'r-')
            plt.show()
            """
            sample_index = 0 
            size_sample = len(Y)
            for idx in range(1, 20):
                if sample_index < size_sample:
                    plt.subplot(4, 5, idx)
                    plt.plot( X[sample_index], color= "red" if Y[sample_index] == 0 else "blue" ) 
                    
                    sample_index = sample_index + 1
    
            plt.show()
            if sample_index >= size_sample:
                break
                
    #debug by samples
    if True:
        sample_index = 0 
        size_sample = len(Y)
        last_shown_index = -1
        show_step = 500
        while(True):
            """
            columns = 10
            rows = 4
            fig, ax_array = plt.subplots(rows, columns,squeeze=False)
            for i,ax_row in enumerate(ax_array):
                for j,axes in enumerate(ax_row):
                    axes.set_title('{},{}'.format(i,j))
                    axes.set_yticklabels([])
                    axes.set_xticklabels([])
                    
            #         axes.plot(you_data_goes_here,'r-')
            plt.show()
            """
            
            #show as much as fit in window
            last_index_covered = -1
            last_class_covered = -1
            idx = 0
            while(True):
                if sample_index < size_sample:
                    if not last_shown_index == -1 and sample_index - last_shown_index < show_step:
                        sample_index = sample_index + 1
                        continue
                
                    if not last_class_covered == -1 and not last_index_covered == sample_index and last_class_covered == Y[sample_index]:
                        sample_index = sample_index + 1
                        continue
                
                    idx = idx + 1
                    last_index_covered = sample_index
                    if last_class_covered == -1:
                        last_class_covered = Y[sample_index]
                        
                    ax = plt.subplot(img_rows, 2, idx)
                    ax.set_title( "idx = " + str(idx) )
                    
                    modr = idx%img_rows
                    if modr == 0:
                        modr = img_rows
                    
                    plt.plot( X[sample_index][ ( img_cols * ( modr - 1 ) ) : ( img_cols * modr ) ], color= "red" if Y[sample_index] == 0 else "blue" ) 
                    
                    if idx % img_rows == 0:
                        last_shown_index = sample_index
                    
                        sample_index = sample_index + 1
                        
                        if not last_class_covered == Y[last_index_covered]:                        
                            break
    
            plt.show()
            if sample_index >= size_sample:
                break
                
    #debug 
    X_cls1_wrong = []
    X_cls2_wrong = []
    
    X_cls1_wrong_prob = []
    X_cls2_wrong_prob = []

    X_cls1_right = []
    X_cls2_right = []
    
    X_cls1_right_prob = []
    X_cls2_right_prob = []
    
    X_cls1_wrong_prob_cnt = {}
    X_cls2_wrong_prob_cnt = {}
    X_cls1_wrong_prob_cnt["09_100"] = 0
    X_cls1_wrong_prob_cnt["08_9"] = 0
    X_cls1_wrong_prob_cnt["07_8"] = 0
    X_cls1_wrong_prob_cnt["06_7"] = 0
    X_cls1_wrong_prob_cnt["05_6"] = 0
    X_cls2_wrong_prob_cnt["09_100"] = 0
    X_cls2_wrong_prob_cnt["08_9"] = 0
    X_cls2_wrong_prob_cnt["07_8"] = 0
    X_cls2_wrong_prob_cnt["06_7"] = 0
    X_cls2_wrong_prob_cnt["05_6"] = 0

    X_cls1_right_prob_cnt = {}
    X_cls2_right_prob_cnt = {}
    X_cls1_right_prob_cnt["09_100"] = 0
    X_cls1_right_prob_cnt["08_9"] = 0
    X_cls1_right_prob_cnt["07_8"] = 0
    X_cls1_right_prob_cnt["06_7"] = 0
    X_cls1_right_prob_cnt["05_6"] = 0
    X_cls2_right_prob_cnt["09_100"] = 0
    X_cls2_right_prob_cnt["08_9"] = 0
    X_cls2_right_prob_cnt["07_8"] = 0
    X_cls2_right_prob_cnt["06_7"] = 0
    X_cls2_right_prob_cnt["05_6"] = 0

    
    X = input_to_be_predicted
    Y = input_to_be_predicted_labels
    
    Y_tmp = res
    Y_tmp_prob = prob_results
    unique, counts = np.unique(input_to_be_predicted_labels, return_counts=True)
    cnts = dict(zip(unique, counts))
    #cnt1 = cnts[0]
    #cnt2 = cnts[1]
    print( cnts )
    #print( "cnt1 " + str(cnt1) + " cnt2 " + str(cnt2) )
    size1 = len(input_to_be_predicted_labels)
    
    #big_class = 0 if cnt1 > cnt2 else 1
    for idx in range(0, size1):
        #TODO temp 
        #size2 = len(X[idx])
        #for idx2 in range(0, size2):
        #    X[idx][idx2] = X[idx][idx2] - 1000 if X[idx][idx2] > 1000 else X[idx][idx2]
    
        #if Y_tmp[idx] == 0 and not Y_tmp[idx] == Y[idx] and Y_tmp_prob[idx][0] >= 0.9999 and len(X_cls1_wrong) < 10000:
        if Y_tmp[idx] == 0 and not Y_tmp[idx] == Y[idx] and len(X_cls1_wrong) < 10000:
            X_cls1_wrong.append(X[idx])
            X_cls1_wrong_prob.append(Y_tmp_prob[idx][0])
            
            if Y_tmp_prob[idx][0] >= 0.9:
                X_cls1_wrong_prob_cnt["09_100"] = X_cls1_wrong_prob_cnt["09_100"] + 1
            elif Y_tmp_prob[idx][0] >= 0.8:
                X_cls1_wrong_prob_cnt["08_9"] = X_cls1_wrong_prob_cnt["08_9"] + 1
            elif Y_tmp_prob[idx][0] >= 0.7:
                X_cls1_wrong_prob_cnt["07_8"] = X_cls1_wrong_prob_cnt["07_8"] + 1
            elif Y_tmp_prob[idx][0] >= 0.6:
                X_cls1_wrong_prob_cnt["06_7"] = X_cls1_wrong_prob_cnt["06_7"] + 1
            elif Y_tmp_prob[idx][0] >= 0.5:
                X_cls1_wrong_prob_cnt["05_6"] = X_cls1_wrong_prob_cnt["05_6"] + 1
            
        #elif Y_tmp[idx] == 1 and not Y_tmp[idx] == Y[idx] and Y_tmp_prob[idx][1] >= 0.9999 and len(X_cls2_wrong) < 10000:
        elif Y_tmp[idx] == 1 and not Y_tmp[idx] == Y[idx] and len(X_cls2_wrong) < 10000:
            X_cls2_wrong.append(X[idx])
            X_cls2_wrong_prob.append(Y_tmp_prob[idx][1])
            
            if Y_tmp_prob[idx][1] >= 0.9:
                X_cls2_wrong_prob_cnt["09_100"] = X_cls2_wrong_prob_cnt["09_100"] + 1
            elif Y_tmp_prob[idx][1] >= 0.8:
                X_cls2_wrong_prob_cnt["08_9"] = X_cls2_wrong_prob_cnt["08_9"] + 1
            elif Y_tmp_prob[idx][1] >= 0.7:
                X_cls2_wrong_prob_cnt["07_8"] = X_cls2_wrong_prob_cnt["07_8"] + 1
            elif Y_tmp_prob[idx][1] >= 0.6:
                X_cls2_wrong_prob_cnt["06_7"] = X_cls2_wrong_prob_cnt["06_7"] + 1
            elif Y_tmp_prob[idx][1] >= 0.5:
                X_cls2_wrong_prob_cnt["05_6"] = X_cls2_wrong_prob_cnt["05_6"] + 1
                
        elif Y_tmp[idx] == 0:
            X_cls1_right.append(X[idx])
            X_cls1_right_prob.append(Y_tmp_prob[idx][0])
            
            if Y_tmp_prob[idx][0] >= 0.9:
                X_cls1_right_prob_cnt["09_100"] = X_cls1_right_prob_cnt["09_100"] + 1
            elif Y_tmp_prob[idx][0] >= 0.8:
                X_cls1_right_prob_cnt["08_9"] = X_cls1_right_prob_cnt["08_9"] + 1
            elif Y_tmp_prob[idx][0] >= 0.7:
                X_cls1_right_prob_cnt["07_8"] = X_cls1_right_prob_cnt["07_8"] + 1
            elif Y_tmp_prob[idx][0] >= 0.6:
                X_cls1_right_prob_cnt["06_7"] = X_cls1_right_prob_cnt["06_7"] + 1
            elif Y_tmp_prob[idx][0] >= 0.5:
                X_cls1_right_prob_cnt["05_6"] = X_cls1_right_prob_cnt["05_6"] + 1
            
        elif Y_tmp[idx] == 1:
            X_cls2_right.append(X[idx])
            X_cls2_right_prob.append(Y_tmp_prob[idx][1])
            
            if Y_tmp_prob[idx][1] >= 0.9:
                X_cls2_right_prob_cnt["09_100"] = X_cls2_right_prob_cnt["09_100"] + 1
            elif Y_tmp_prob[idx][1] >= 0.8:
                X_cls2_right_prob_cnt["08_9"] = X_cls2_right_prob_cnt["08_9"] + 1
            elif Y_tmp_prob[idx][1] >= 0.7:
                X_cls2_right_prob_cnt["07_8"] = X_cls2_right_prob_cnt["07_8"] + 1
            elif Y_tmp_prob[idx][1] >= 0.6:
                X_cls2_right_prob_cnt["06_7"] = X_cls2_right_prob_cnt["06_7"] + 1
            elif Y_tmp_prob[idx][1] >= 0.5:
                X_cls2_right_prob_cnt["05_6"] = X_cls2_right_prob_cnt["05_6"] + 1                    
    
    print( "X_cls1_wrong " + str(len(X_cls1_wrong)) )
    print( "X_cls2_wrong " + str(len(X_cls2_wrong)) )
    print( "X_cls1_wrong_prob_cnt " + str(X_cls1_wrong_prob_cnt) )
    print( "X_cls2_wrong_prob_cnt " + str(X_cls2_wrong_prob_cnt) )
    print( "X_cls1_right_prob_cnt " + str(X_cls1_right_prob_cnt) )
    print( "X_cls2_right_prob_cnt " + str(X_cls2_right_prob_cnt) )
    if len(X_cls1_wrong) > 0 or len(X_cls2_wrong) > 0 or len(X_cls1_right) > 0 or len(X_cls1_right) > 0:
        sizeloop = len(X_cls1_wrong) if len(X_cls1_wrong) > len(X_cls2_wrong) else len(X_cls2_wrong)
        sizeloop = len(X_cls1_right) if len(X_cls1_right) > sizeloop else sizeloop
        sizeloop = len(X_cls2_right) if len(X_cls2_right) > sizeloop else sizeloop
        
        for idx in range(0, sizeloop):
            is_plot = False
        
            if len(X_cls1_wrong) > idx:
                print( "X_cls1_wrong prob " + str(X_cls1_wrong_prob[idx]) )
                
                is_plot = True
                plt.subplot(2, 2, 1)
                plt.plot( X_cls1_wrong[idx], color="red") 

            if len(X_cls2_wrong) > idx:
                print( "X_cls2_wrong prob " + str(X_cls2_wrong_prob[idx]) )
                
                is_plot = True
                plt.subplot(2, 2, 2)
                plt.plot( X_cls2_wrong[idx], color="blue") 
                    
            if len(X_cls1_right) > idx:
                print( "X_cls1_right prob " + str(X_cls1_right_prob[idx]) )
                
                is_plot = True
                plt.subplot(2, 2, 3)
                plt.plot( X_cls1_right[idx], color="red") 

            if len(X_cls2_right) > idx:
                print( "X_cls2_right prob " + str(X_cls2_right_prob[idx]) )
                
                is_plot = True
                plt.subplot(2, 2, 4)
                plt.plot( X_cls2_right[idx], color="blue")                         

            if is_plot == True:
                plt.show()        
