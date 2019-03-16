'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import keras
from keras.models import load_model

import sys, os
import json
from numpy import array
import numpy as np

ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

to_be_predicted_index = -1 


n_classes_ = int(sys.argv[4]) if len(sys.argv) >= 5 else 3

input_to_be_predicted = np.array( json.load( open( sys.argv[3] ) ) )
input_to_be_predicted_labels = np.array( json.load( open( sys.argv[14] ) ) ) if len(sys.argv) >= 15 else None

outfile_path = sys.argv[15] if len(sys.argv) >= 16 else None
outfile_path_prob = sys.argv[19] if len(sys.argv) >= 20 else None

rows = int(sys.argv[16]) if len(sys.argv) >= 17 else 0
cols = int(sys.argv[17]) if len(sys.argv) >= 18 else 0

modelfile_path = sys.argv[18] if len(sys.argv) >= 19 else None

is_sample_debug_only = int(sys.argv[20]) if len(sys.argv) >= 21 else 0


#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

model = None

import os.path
if os.path.isfile(modelfile_path): 
    model = load_model(modelfile_path)    
    
# input image dimensions
img_rows, img_cols = rows, cols   #28, 28

x_test = input_to_be_predicted


if is_sample_debug_only == 1 or model == None:    
        #
        total_input_files = int(sys.argv[21]) if len(sys.argv) >= 22 else 0
        X = []
        Y = []
        for i in range(0, total_input_files):
            if i == 0:
                X = array( json.load( open( sys.argv[2].replace('{i}', str(i)) ) ) ) 
                Y = array( json.load( open( sys.argv[8].replace('{i}', str(i)) ) ) ) 
            else:
                X = np.concatenate( ( X, array( json.load( open( sys.argv[2].replace('{i}', str(i)) ) ) ) ), axis=0 )
                Y = np.concatenate( ( Y, array( json.load( open( sys.argv[8].replace('{i}', str(i)) ) ) ) ), axis=0 )
    
        print("X length " + str(len(X)) )
        print("X[0] length " + str(len(X[0])) )
        print("Y length " + str(len(Y)) )
        unique, counts = np.unique(Y, return_counts=True)
        print( dict(zip(unique, counts)) )
        print(Y)

    
if is_sample_debug_only == 0:    

    num_classes = n_classes_ #10
    if model == None:    

        batch_size = 128
        epochs = 12    #1000     #12

        is_use_sample_data = False
        if is_use_sample_data == True:
            df = pd.read_csv( ABS_PATh + 'input/spam.csv',delimiter=',',encoding='latin-1')
            df.head()
            
            df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
            df.info()
            
            #Understand the distribution better.
            sns.countplot(df.v1)
            plt.xlabel('Label')
            plt.title('Number of ham and spam messages')
            plt.show()

            X = df.v2
            Y = df.v1
            
            le = LabelEncoder()
            Y = le.fit_transform(Y)
            Y = Y.reshape(-1,1)
        else:
            # convert class vectors to binary class matrices
            Y = keras.utils.to_categorical(Y, num_classes)
            
        
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
                
        #X_train = X_train.astype('float32')
        #X_test = X_test.astype('float32')


        max_words = 1000
        max_len = 150
        if is_use_sample_data == True:
            tok = Tokenizer(num_words=max_words)
            tok.fit_on_texts(X_train)
            sequences = tok.texts_to_sequences(X_train)
            sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
            
            test_sequences = tok.texts_to_sequences(X_test)
            test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
        else:
            max_words = 500
            max_len = len( X_train[0] )
            sequences_matrix = X_train
            test_sequences_matrix = X_test
            
        print( sequences_matrix[0] )
        print( type( sequences_matrix[0] ) )
        print( sequences_matrix[0].shape )
        print( sequences_matrix )
        
        def RNN():
            inputs = Input(name='inputs',shape=[max_len])
            layer = Embedding(max_words,50,input_length=max_len)(inputs)
            layer = LSTM(64)(layer)
            layer = Dense(256,name='FC1')(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.5)(layer)
            #layer = Dense(1,name='out_layer')(layer)
            layer = Dense(2,name='out_layer')(layer)
            #layer = Activation('sigmoid')(layer)
            layer = Dense(2,name='out_layer2', activation='softmax')(layer)
            model = Model(inputs=inputs,outputs=layer)
            return model

        ##sample_weights
        #from sklearn.utils import class_weight
        #list_classes = [0, 1]
        #y = X_train[list_classes].values
        #sample_weights = class_weight.compute_sample_weight('balanced', y)
        
        model = RNN()
        model.summary()
        #model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        
        model.fit(sequences_matrix,Y_train,batch_size=128,epochs=epochs,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.001)])     #,sample_weight=sample_weights,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
          
        
        accr = model.evaluate(test_sequences_matrix,Y_test)
        
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

        model.save( modelfile_path )
        
    else:
        tmp = ''
        
        #x_test = x_test.astype('float32')

        ##TODO temp
        ## convert class vectors to binary class matrices
        #itbplc = keras.utils.to_categorical(input_to_be_predicted_labels, num_classes)
        #accr = model.evaluate(x_test,itbplc)
        #print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        

    #input_to_be_predicted = x_test    

    print("target")
    print( input_to_be_predicted_labels )

    prob_results = model.predict( x_test )
    res = prob_results.argmax(axis=-1)
    print("result")
    unique, counts = np.unique(res, return_counts=True)
    print( dict(zip(unique, counts)) )
    class_0_prob, class_1_prob = zip(*prob_results)
    class_0_prob = np.array( class_0_prob )
    class_1_prob = np.array( class_1_prob )
    print( "class 0 prob > 0.9 " + str( ( (0.9 < class_0_prob) ).sum() ) )
    print( "class 0 prob > 0.8 & <= 0.9 " + str( ((0.8 < class_0_prob) & (class_0_prob <= 0.9)).sum() ) )
    print( "class 0 prob > 0.7 & <= 0.8 " + str( ((0.7 < class_0_prob) & (class_0_prob <= 0.8)).sum() ) )
    print( "class 0 prob > 0.6 & <= 0.7 " + str( ((0.6 < class_0_prob) & (class_0_prob <= 0.7)).sum() ) )
    print( "class 0 prob > 0.5 & <= 0.6 " + str( ((0.5 < class_0_prob) & (class_0_prob <= 0.6)).sum() ) )
    print( "class 1 prob > 0.9 " + str( ( (0.9 < class_1_prob) ).sum() ) )
    print( "class 1 prob > 0.8 & <= 0.9 " + str( ((0.8 < class_1_prob) & (class_1_prob <= 0.9)).sum() ) )
    print( "class 1 prob > 0.7 & <= 0.8 " + str( ((0.7 < class_1_prob) & (class_1_prob <= 0.8)).sum() ) )
    print( "class 1 prob > 0.6 & <= 0.7 " + str( ((0.6 < class_1_prob) & (class_1_prob <= 0.7)).sum() ) )
    print( "class 1 prob > 0.5 & <= 0.6 " + str( ((0.5 < class_1_prob) & (class_1_prob <= 0.6)).sum() ) )
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
    if False:
        sample_index = 0 
        size_sample = len(Y)
        last_shown_index = -1
        show_step = 500  #500
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
                    
                    print( X[sample_index][ ( img_cols * ( modr - 1 ) ) : ( img_cols * modr ) ] )
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
