from utils import *

#Parameters to change
min_count = 1
size_of_layers = 10
n_workers = 1
sentence_file_path = 'sentences.txt'

#train and save the model
get_word2vec_model(sentence_file_path, min_count, size_of_layers, n_workers)

