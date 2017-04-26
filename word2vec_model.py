from utils import *

#Parameters to change
min_count = 1
size_of_layers = 5
n_workers = 1
sentences_path = 'sentences.txt'
model_path_save = 'trained_models/word2vec_model_100'

sentences = gensim.models.word2vec.LineSentence(sentences_path)
print "training model"
word2vec_model = gensim.models.Word2Vec(sentences, min_count = 1, size = size_of_layers, workers = n_workers)
word2vec_model.save(model_path_save)

