# Shapiro-Wilk Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro

import numpy as np
from numpy import array
import os, sys
import json

# Parse options
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='a utility')

parser.add_argument('-b', '--is_debug', action='store_true', help='A boolean True False')
parser.add_argument('-s', '--is_use_sample_data', action='store_true', help='A boolean True False')
parser.add_argument('-t', '--total_input_files', type=int, nargs='?', help='total_input_files')
parser.add_argument('-i', '--input_file', type=str, nargs='?', help='input_file')
parser.add_argument('-l', '--input_labels_file', type=str, nargs='?', help='input_labels_file')
parser.add_argument('-a', '--is_apply_labels', action='store_true', help='label class will be applied to first element of record, so whatever there would be overwritten')
parser.add_argument('-m', '--class_multiplier', type=float, nargs='?', help='label class value multiplier to increase or decrease numerical importance of class label')
parser.add_argument('--is_apply_pca_first', action='store_true', help='')
parser.add_argument('--pca_components', type=int, nargs='?', help='')


FLAGS = parser.parse_args()
print(FLAGS)

if FLAGS.is_use_sample_data:
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data = 5 * randn(100) + 50
else:
    data = []
    input_labels = []
    print("total_input_files")
    print(FLAGS.total_input_files)
    for i in range(0, FLAGS.total_input_files):
        print("total_input_files i " + str(i))
        if i == 0:
            data = array( json.load( open( FLAGS.input_file.replace('{i}', str(i)) ) ) ) 
            input_labels = array( json.load( open( FLAGS.input_labels_file.replace('{i}', str(i)) ) ) ) 
        else:
            data = np.concatenate( ( data, array( json.load( open( FLAGS.input_file.replace('{i}', str(i)) ) ) ) ), axis=0 )
            input_labels = np.concatenate( ( input_labels, array( json.load( open( FLAGS.input_labels_file.replace('{i}', str(i)) ) ) ) ), axis=0 )
    
    print(data.shape)
    data = data.reshape((data.shape[0], -1))
    print(data.shape)
    
    if FLAGS.is_apply_labels:
        if FLAGS.class_multiplier == None:
            data[:,0] = input_labels[:]
        else:
            data[:,0] = input_labels[:] * FLAGS.class_multiplier
            
    #pca 
    if FLAGS.is_apply_pca_first:
    
        print(data.shape)
        
        from sklearn.decomposition import PCA
        p = PCA(n_components = 1600).fit_transform(data)
        print( type(p) )
        print( p )
        print( p.shape )
    

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('shapiro test. Sample looks Gaussian (fail to reject H0)')
else:
    print('shapiro test. Sample does not look Gaussian (reject H0)')

    
from scipy.stats import normaltest
# normality test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('scipy.stats.normaltest test. Sample looks Gaussian (fail to reject H0)')
else:
	print('scipy.stats.normaltest test. Sample does not look Gaussian (reject H0)')    

    
from scipy.stats import anderson
# normality test
result = anderson(data)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('anderson test. %.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('anderson test. %.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))    