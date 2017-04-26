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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print('Prediction === ', y_pred)
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_text_network_lstm(input_shape, input_shape_time_dist):
	kernel_size = (5,5)
	embedding_size = 128
	model = Sequential()

	model.add(TimeDistributed(Dense(embedding_size, activation='relu', kernel_regularizer=regularizers.l2(0.005), input_shape=input_shape), input_shape=input_shape_time_dist))
	model.add(LSTM(embedding_size, return_sequences=False))
	model.add(Dense(embedding_size, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
	#model.add(TimeDistributed(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape_conv),	input_shape=input_shape_time_dist))
	#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	#model.add(TimeDistributed(Conv2D(64, kernel_size=kernel_size, padding='same', activation='relu')))
	#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	#model.add(TimeDistributed(Flatten()))
	#model.add(TimeDistributed(Dense(embedding_size, kernel_regularizer=regularizers.l2(0.005))))
	#model.add(GlobalAveragePooling1D())
	
	return model

def create_base_image_network(conv_input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    kernel_size = (5,5)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=conv_input_shape))
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    return model

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    mean = np.mean(predictions, axis=0)
    return labels[predictions.ravel() < mean].mean()

#main
epochs = 10
val_split = 0.1

dataset = Dataset()
I, T, y = dataset.load_pairs()

numb_examples, rows, cols, channels = I.shape

if K.image_data_format() == 'channels_first':
    I = I.reshape(numb_examples, channels, rows, cols)
    input_shape_img = (3, rows, cols)
else:
    I = I.reshape(numb_examples, rows, cols, channels)
    input_shape_img = (rows, cols, channels)

I = I.astype('float32')
I /= 255  

T = T.astype('float32')
numb_examples, numb_words, word_emb = T.shape
input_shape_text = (word_emb, )
input_shape_text_time_distributed = (numb_words, word_emb)

base_image_network = create_base_image_network(input_shape_img)
base_text_network = create_base_text_network_lstm(input_shape_text, input_shape_text_time_distributed)

#create_base_network(input_shape_conv, input_shape_time_dist)

input_i = Input(shape=input_shape_img)
input_t = Input(shape=input_shape_text_time_distributed)

processed_i = base_image_network(input_i)
processed_t = base_text_network(input_t)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_i, processed_t])

model = Model([input_i, input_t], distance)

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([I, T], y, batch_size=128, epochs=epochs, validation_split=val_split)

pred = model.predict([I, T])
tr_acc = compute_accuracy(pred, y)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

#model.save_weights("eucledean_distance_model.h5")

#Saving Model:
save_keras_mode('distance_lstm', model)
