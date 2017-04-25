from utils import *

#testing getting saved model:
#model_path_save = '/Users/andymartinez/git/Image-Retrival-From-Text/trained_models/word2vec_model_100'
sentence = "a vandalized stop sign and a red beetle on the road"
#sentence_size = 50
#new_model = gensim.models.Word2Vec.load(model_path_save)
padded_sentence = get_sentence_encoding(sentence)
print padded_sentence

print "everything good so far"