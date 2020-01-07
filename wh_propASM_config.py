"""
2020. 01. 04 
Follow up code for
P. Chakravarthula et al., "Wirtinger Holography for Near-Eye Displays", 
SIGGRAPH ASIA, ACM TOG, 2019

by Dongheon Yoo

Optimizing phase hologram via gradient descent algorithm
Use Tensorflow 2.0
"""

import os

# Basic parameters
PRJ_NAME = ('WH')
ROOT = os.path.join(os.getcwd(), PRJ_NAME)
RESULT_DIR = os.path.join(ROOT, 'result')

PARAM = {'SLM_HEIGHT' : 1024,
         'SLM_WIDTH' : 1024,
         'PROPAGATION_DISTANCE' : 15e-3,
         'PIXEL_PITCH' : 6.4e-6,
         'WAVELENGTH' : 520e-9,
         'LEARNING_RATE' : 1e-1, # Learning rate
         'BATCH_SIZE' : 1,
         'EPOCH' : 100,  # The number of training epoch
         'STEPS_PER_PLOT_IMAGE' : 1,
         'PRJ_NAME' : PRJ_NAME,
         'ROOT' : ROOT,
         'RESULT_DIR' : RESULT_DIR
         }
