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
    

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


    