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

#main ----------------------------------------
numb_frames = 500
numb_ranks = 100

model_name = ['trained_models/distance_lstm_avg_cross_final'][0]
output_name = model_name + '_ranks.mat'
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
frames = dataset.get_test_frames()[0:numb_frames]
ranks = np.zeros(numb_ranks)

h = 0
for query_frame in frames:
	print('h = ', h)
	h+=1
	caption = query_frame.get_captions_embeding()[0]
	#print('\n==============================================================')
	#print('Query Frame =', query_frame.get_id())

	distances = {}
	for galery_frame in frames:
		img = galery_frame.get_image()
		dist = model.predict([array([img]), array([caption])])[0][0]
		#print('Gallery Frame =', galery_frame.get_id())
		#print('Distance = ', dist)
		if dist in distances:
			distances[dist].append(galery_frame.get_id())
		else:
			distances[dist] = [galery_frame.get_id()]
		
		 
	keys = distances.keys()
	keys.sort()

	for i in range(len(keys)):
		if query_frame.get_id() in distances[keys[i]]:
			for j in range(i, len(ranks)):
				ranks[j] += 1

for i in range(len(ranks)):
	ranks[i] = (ranks[i]*1.0)/numb_frames
	

print('================================================================')
print('RANKS:')
print(ranks)  	
ranks = np.array(ranks, dtype=np.object)
scipy.io.savemat(output_name, mdict={'ranks': ranks})



