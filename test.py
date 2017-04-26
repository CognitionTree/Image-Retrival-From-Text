from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, LSTM, GlobalAveragePooling1D
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

def contrastive_loss(y_true, y_pred):
	'''
	Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	
	print('Prediction === ', y_pred)
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
	#Compute classification accuracy with a fixed threshold on distances.
	mean = np.mean(predictions, axis=0)
	return labels[predictions.ravel() < mean].mean()

d = Dataset()
I, T, y = d.load_pairs(False)

json_file = open('trained_models/distance_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('trained_models/distance_lstm.h5')

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
#model.fit([I1, I2], y, batch_size=128, epochs=epochs, validation_split=val_split)

pred = model.predict([I, T])
te_acc = compute_accuracy(pred, y)
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))