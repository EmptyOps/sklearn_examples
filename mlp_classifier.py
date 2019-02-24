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

outfile_path = sys.argv[15] if len(sys.argv) >= 16 else None

if __name__ == '__main__':
    print("X length " + str(len(X)) )
    print("X[0] length " + str(len(X[0])) )
    print("Y length " + str(len(Y)) )
    print(Y)

    
    #apply PCA only if needed 
    is_do_pca = False
    if is_do_pca == True:
        from sklearn.decomposition import PCA
        import math
        n_components_val = math.ceil( len(X[0]) / 5 )
        print( X.shape )
        X = PCA( n_components= n_components_val ).fit_transform( X )
        print( X.shape )
    
    #X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    y = Y #[0, 0, 0, 1]
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-1, activation='identity', verbose=False,
    #                    tol=1e-20, max_iter=10000, 
    #                    hidden_layer_sizes=(25, 25, 25, 25), random_state=1)
    
    loss_curve_ = []
    clf = MLPClassifier(solver='lbfgs', alpha=1e-1, activation='identity', verbose=True,
                        tol=1e-20, max_iter=1, warm_start=True, 
                        hidden_layer_sizes=(288, 144, 72, 36, 18, 8, 2), random_state=1)
                        
    for i in range(1, 5000):
        clf.fit(X, y)
        loss_curve_.append( clf.loss_ )

        if ENV == 1 and i % 1000 == 0:
            from matplotlib import pyplot as plt
            
            #loss curve 
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Loss curve")
            plt.plot( loss_curve_ )
            plt.show()
        

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

    
    #save model 
    #from joblib import dump, load
    #dump(clf, 'filename.joblib') 
    import pickle
    s = pickle.dumps(clf)
    print(s)
    
    
    #apply PCA only if needed 
    if is_do_pca == True:
        print( input_to_be_predicted.shape )
        input_to_be_predicted = PCA(n_components=n_components_val).fit_transform( input_to_be_predicted )
        print( input_to_be_predicted.shape )
    
    print("target")
    print( input_to_be_predicted_labels )

    res = clf.predict( input_to_be_predicted )
    
    print("result")
    print(res)                      
                          
                          
    prob_results = clf.predict_proba( input_to_be_predicted )
    print(prob_results)
                      
    if not outfile_path == None:
        with open( outfile_path, 'w') as outfile:
            json.dump(prob_results.tolist(), outfile)                      
        
        
    # in dev mode 
    if ENV == 1:
        from matplotlib import pyplot as plt
        
    
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
                    if max( X_cls1_wrong[idx] ) > 6:
                        print( "X_cls1_wrong prob " + str(X_cls1_wrong_prob[idx]) )
                        
                        is_plot = True
                        plt.subplot(2, 2, 1)
                        plt.plot( X_cls1_wrong[idx], color="red") 

                if len(X_cls2_wrong) > idx:
                    if max( X_cls2_wrong[idx] ) > 6:
                        print( "X_cls2_wrong prob " + str(X_cls2_wrong_prob[idx]) )
                        
                        is_plot = True
                        plt.subplot(2, 2, 2)
                        plt.plot( X_cls2_wrong[idx], color="blue") 
                        
                if len(X_cls1_right) > idx:
                    if max( X_cls1_right[idx] ) > 6:
                        print( "X_cls1_right prob " + str(X_cls1_right_prob[idx]) )
                        
                        is_plot = True
                        plt.subplot(2, 2, 3)
                        plt.plot( X_cls1_right[idx], color="red") 

                if len(X_cls2_right) > idx:
                    if max( X_cls2_right[idx] ) > 6:
                        print( "X_cls2_right prob " + str(X_cls2_right_prob[idx]) )
                        
                        is_plot = True
                        plt.subplot(2, 2, 4)
                        plt.plot( X_cls2_right[idx], color="blue")                         

                if is_plot == True:
                    plt.show()        
                    
                    