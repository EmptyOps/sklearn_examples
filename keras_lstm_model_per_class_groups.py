'''
- file path argumnets should have {icls} markup to let model differentiate over class groups 
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
import gc
import json
from numpy import array
import numpy as np
import random

ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

to_be_predicted_index = -1 


n_classes_ = int(sys.argv[4]) if len(sys.argv) >= 5 else 3
base_classes_  = int(sys.argv[22]) if len(sys.argv) >= 23 else 0

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

is_debug = True
is_evaluation_mode = False
if os.path.isfile(modelfile_path.replace( '{icls}', '1' )): 
    model = load_model( modelfile_path.replace( '{icls}', '1' ) ) 
    if not model == None:
        is_evaluation_mode = True
    
# input image dimensions
img_rows, img_cols = rows, cols   #28, 28

x_test = input_to_be_predicted
y_test = None


#
if is_debug:
    print( "n_classes_ " + str(n_classes_) )

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

def wait_for_user( msg ):
    # raw_input returns the empty string for "enter"
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}

    print( msg )
    choice = input().lower()
    if choice in no:
       exit()
    elif not (choice in yes):
       print("Please respond with 'yes' or 'no'")
       wait_for_user()
    
def randomize( a, b ):
    #a = ["Spears", "Adele", "NDubz", "Nicole", "Cristina"]
    #b = [1, 2, 3, 4, 5]

    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    
    return a, b

def make_equal( a, b, cls1val, cls2val ):
    cnts = {}
    cnts[cls1val] = 0
    cnts[cls2val] = 0

    c = []
    d = []
    sizel = len(a)
    for i in range(0, sizel):

        if b[i] == cls2val or cnts[cls1val] < cnts[cls2val] + 100:
            c.append( a[i] )
            d.append( b[i] )
            
            cnts[b[i]] = cnts[b[i]] + 1
        else:
            continue
    
    return c, d
    
    
#    
icls = {}

sizex = 0
Xs = {}
Ys = {}

input_to_be_predictedXs = {}
input_to_be_predictedYs = {}
res_all = {}
prob_results_all = {}

import math
for i in range(0, n_classes_):
    icls[i] = int( math.ceil( (i+1) / 2 ) )
    Xs[ icls[i] ] = []
    Ys[ icls[i] ] = []
    input_to_be_predictedXs[ icls[i] ] = []
    input_to_be_predictedYs[ icls[i] ] = []

if is_sample_debug_only == 1 or model == None:    
        #
        print( "(!) Merging inputs, should only be executed in training mode." )
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
        
        sizex = len(X)
        for i in range(0, sizex):
            Xs[ icls[Y[i]] ].append( X[i] )
            Ys[ icls[Y[i]] ].append( Y[i] % 2 )
            
        #
        Xs[ 1 ], Ys[ 1 ] = randomize( Xs[ 1 ], Ys[ 1 ] )
        #Xs[ 1 ], Ys[ 1 ] = make_equal( Xs[ 1 ], Ys[ 1 ], 0, 1 )
        Xs[ 2 ], Ys[ 2 ] = randomize( Xs[ 2 ], Ys[ 2 ] )
        #Xs[ 2 ], Ys[ 2 ] = make_equal( Xs[ 2 ], Ys[ 2 ], 0, 1 )
            
sizetestx = len(x_test)
for i in range(0, sizetestx):
    input_to_be_predictedXs[ icls[input_to_be_predicted_labels[i]] ].append( x_test[i] )
    input_to_be_predictedYs[ icls[input_to_be_predicted_labels[i]] ].append( input_to_be_predicted_labels[i] % 2 )
            
            
for i in range(0, n_classes_):
    if (i+1) % 2 == 0:
        if len( input_to_be_predictedYs[ icls[i] ] ) > 0:
            print("processing icls " + str(icls[i]))
        else:
            print("skipping icls " + str(icls[i]))
            continue

        model = None
        model_no = icls[i] if n_classes_ <= base_classes_ else ( base_classes_/2 if icls[i] % (base_classes_/2) == 0 else icls[i] % (base_classes_/2) )
        model_no = int( model_no )
        if os.path.isfile(modelfile_path.replace( '{icls}', str( model_no ) )): 
            model = load_model( modelfile_path.replace( '{icls}', str( model_no ) ) )    

        print("model_no " + str(model_no) + " base_classes_ " + str(base_classes_) + " ")
            
        if is_sample_debug_only == 0:    

            #
            if is_evaluation_mode == True and model == None:
                print( "Fatal error! Model not found in evauation mode" )
                sdfkjhkdsjfhkjdshf
        
            num_classes = 2 #n_classes_ #10
            if model == None:    
                X = array( Xs[ icls[i] ] )
                Y = array( Ys[ icls[i] ] )
                
                print("X length " + str(len(X)) )
                print("X[0] length " + str(len(X[0])) )
                print("Y length " + str(len(Y)) )
                unique, counts = np.unique(Y, return_counts=True)
                print( dict(zip(unique, counts)) )
            

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
                #print( sequences_matrix )
                
                ##sample_weights
                #from sklearn.utils import class_weight
                #list_classes = [0, 1]
                #y = X_train[list_classes].values
                #sample_weights = class_weight.compute_sample_weight('balanced', y)
                
                model = RNN()
                model.summary()
                #model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

                
                last_loss = 1
                for epch in range(0, 1): #epochs):
                    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=epochs,   #1, warm_start=True,  #epochs
                      validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.001)])     #,sample_weight=sample_weights,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]

                    accr = model.evaluate(test_sequences_matrix,Y_test)
                    
                    model.save( modelfile_path.replace( '{icls}', str(icls[i]) + '-' + str(epch) + '-{:0.3f}-{:0.3f}'.format(accr[0],accr[1]) ) ) 
                    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))                
                
                    if accr[0] >= last_loss - 0.001:
                        break
                
                    last_loss = accr[0]
                    
                
            else:
                tmp = ''
                
                #x_test = x_test.astype('float32')

                ##TODO temp
                ## convert class vectors to binary class matrices
                #itbplc = keras.utils.to_categorical(input_to_be_predicted_labels, num_classes)
                #accr = model.evaluate(x_test,itbplc)
                #print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
                
                x_test = array( input_to_be_predictedXs[ icls[i] ] )
                y_test = array( input_to_be_predictedYs[ icls[i] ] )
                

            #input_to_be_predicted = x_test    

            try:
                if is_debug:
                    print("input")
                    print( x_test.shape )
                    print( x_test )
                    print("target")
                    print( y_test )

                prob_results = model.predict( x_test )
                res = prob_results.argmax(axis=-1)
                unique, counts = np.unique(res, return_counts=True)
                
                if is_debug:
                    print("result")
                    print( dict(zip(unique, counts)) )
                    
                class_0_prob, class_1_prob = zip(*prob_results)
                class_0_prob = np.array( class_0_prob )
                class_1_prob = np.array( class_1_prob )
                
                if is_debug:
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
                
                res_all[icls[i]] = res 
                prob_results_all[icls[i]] = prob_results 
            except Exception as e:
                print( e )
                
            msg = "processing icls " + str(icls[i]) + " is done, press enter to continue... "
            if ENV < 1:
                wait_for_user( msg ) 
            else:
                print( msg )


#          
res = []
prob_results = []  
cnts = {}
for i in range(0, sizetestx):
    iclsn = icls[input_to_be_predicted_labels[i]]
    if not iclsn in cnts:
        cnts[iclsn] = 0
        
    #print( "res i " + str(i) + " iclsn " + str(iclsn) + " cnts[iclsn] " + str(cnts[iclsn]) + " prob_results " )
    res.append( res_all[ iclsn ][ cnts[iclsn] ] )
    prob_results.append( prob_results_all[ iclsn ][ cnts[iclsn] ] )
    cnts[iclsn] = cnts[iclsn] + 1
    
res = array( res ) 
prob_results = array( prob_results ) 
    
del res_all
del prob_results_all
gc.collect()
            
            
if not outfile_path == None:
    with open( outfile_path, 'w') as outfile:
        json.dump(res.tolist(), outfile)                      

if not outfile_path_prob == None:
    with open( outfile_path_prob, 'w') as outfile:
        json.dump(prob_results.tolist(), outfile)                      
                
            
if is_evaluation_mode == False:
    for i in range(0, n_classes_):
        if (i+1) % 2 == 0:
            X = array( Xs[ icls[i] ] )
            Y = array( Ys[ icls[i] ] )
            
            x_test = array( input_to_be_predictedXs[ icls[i] ] )
            y_test = array( input_to_be_predictedYs[ icls[i] ] )
                
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

                
                X = x_test
                Y = y_test
                
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
