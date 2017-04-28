from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Merge, Average
from keras import backend as K
from keras.utils import plot_model
from keras import metrics
from keras import regularizers
from keras.models import model_from_yaml
from keras.models import model_from_json
import os

from Dataset import *
from utils import *
import scipy

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print('Prediction === ', y_pred)
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    mean = np.mean(predictions, axis=0)
    return labels[predictions.ravel() < mean].mean()

def compute_avg_distances(predictions, labels):
	same_avg = predictions[labels.ravel() == 1].mean()
	diff_avg = predictions[labels.ravel() == 0].mean()
	return same_avg, diff_avg 

def save_results(is_training=True):
	model_name = ['trained_models/distance_lstm_avg_cross_final'][0]
	madel_path = model_name+'.json'
	weights_path = model_name+'.h5'

	#Load Model
	json_file = open(madel_path, 'r')
	model = json_file.read()
	json_file.close()
	model = model_from_json(model)

	# load weights into new model
	model.load_weights(weights_path)

	rms = RMSprop()
	model.compile(loss=contrastive_loss, optimizer=rms)


	#Predict Training
	dataset = Dataset()
	I, T, y = dataset.load_pairs(is_training)

	pred = model.predict([I, T])
	if len(pred) == 2:
		pred = pred[0]

	te_acc = compute_accuracy(pred, y)
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

	same_dist, diff_dist = compute_avg_distances(pred, y)
	print('Avg = ', (same_dist + diff_dist)/2)
	print('Distance of positive pairs = ', same_dist)
	print('Distance of negative pairs = ', diff_dist)
	print('Difference = ', diff_dist - same_dist)

	distances = np.array(pred, dtype=np.object)
	labels = np.array(y, dtype=np.object)
		

	output_name = model_name + '_testing.mat'
	if is_training:
		output_name = model_name + '_training.mat' 

	scipy.io.savemat(output_name, mdict={'distance': distances, 'label':labels, 'pos_avg_distance':np.array([same_dist], dtype=np.object), 'neg_avg_distance':np.array([diff_dist], dtype=np.object), 'accuracy':np.array([te_acc], dtype=np.object)})
	
save_results(is_training=True)
save_results(is_training=False)	
#Predict Testing
