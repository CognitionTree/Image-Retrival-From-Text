import gensim
from train_word2vec import *
from numpy import *

def get_word2vec_model(sentences_path = 'sentences.txt', min_count = 1, size_of_layers = 100, n_workers = 1):
	sentences = gensim.models.word2vec.LineSentence(sentences_path)
	word2vec_model = gensim.models.Word2Vec(sentences, min_count = 1, size = size_of_layers, workers = n_workers)
	word2vec_model.save('trained_models/word2vec_model')

def get_sentence_encoding(model, sentence, sentence_size):
	sentence_emb = model[sentence.split()]
	n_words = len(sentence_emb)
	embedding_size = len(sentence_emb[0])
	
	padding_size = sentence_size - n_words
	padding = zeros((padding_size, embedding_size))
	padded_sentence = append(padding, sentence_emb, axis=0)
	
	return padded_sentence
	
