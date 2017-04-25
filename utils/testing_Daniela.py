from train_word2vec import *

#playing with parameters for get_word2vec_model
all_sentences = [['first', 'sentence'], ['second', 'sentence']]
min_count = 1
size_of_layers = 5
n_workers = 1

#testing get_word2vec_model
#get_word2vec_model('sentences.txt', min_count, size_of_layers, n_workers)


#testing getting saved model:
model_path = '/Users/danielaflorit/Github/Image-Retrival-From-Text/trained_models/word2vec_model'
new_model = gensim.models.Word2Vec.load(model_path)
#print new_model['first']
sentence = "first second"
sentence_size = 5
padded_sentence = get_sentence_encoding(new_model, sentence, sentence_size)

print padded_sentence
print "everything good so far"