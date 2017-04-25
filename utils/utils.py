import gensim
from numpy import *
from train_word2vec import *
from nltk.tokenize import word_tokenize

def tokenize_one_sentence(sentence):
	sentence = word_tokenize(sentence)
	tokenized_sentence = []
	for word in sentence:
		word = word.lower()
		if word.isalnum():
			tokenized_sentence.append(word)
	return tokenized_sentence

def tokenize_all_sentences(sentences):
	sentence_array = []
	i = 0
	for sentence in sentences:
		print i, " out of ", len(sentences)	
		tokenized_sentence = tokenize_one_sentence(sentence)
		sentence_array.append(tokenized_sentence)
		i += 1
	return sentence_array
		
def get_word2vec_model(sentences_path = 'sentences.txt', min_count = 1, size_of_layers = 100, n_workers = 1):
	model_path_save = '/Users/andymartinez/git/Image-Retrival-From-Text/trained_models/word2vec_model_100'
	sentences = gensim.models.word2vec.LineSentence(sentences_path)
	print "training model"
	word2vec_model = gensim.models.Word2Vec(sentences, min_count = 1, size = size_of_layers, workers = n_workers)
	word2vec_model.save(model_path_save)

def get_sentence_encoding(model, sentence, sentence_size):
	sentence_emb = model[sentence.split()]
	n_words = len(sentence_emb)
	embedding_size = len(sentence_emb[0])
	
	padding_size = sentence_size - n_words
	padding = zeros((padding_size, embedding_size))
	padded_sentence = append(padding, sentence_emb, axis=0)
	
	return padded_sentence

