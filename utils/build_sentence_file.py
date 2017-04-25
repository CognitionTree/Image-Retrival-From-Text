import json
from utils import *

def parse_json_file(file_path):
	sentences = []
	file = json.loads(open(file_path).read())
	captions = file['annotations']
	for caption in captions:
		sentence = str(caption['caption'])
		sentences.append(sentence)
	return sentences

def write_sentence_file(sentence_array, sentences_path):
	file = open(sentences_path, 'w')
	for sentence in sentence_array:
		sentence = ' '.join(sentence) + '\n'
		file.write(sentence)
	file.close()
		
	

#-----Main----
file_path_train = '/Users/andymartinez/git/Image-Retrival-From-Text/dataset/annotations/captions_train2014.json'
file_path_val = '/Users/andymartinez/git/Image-Retrival-From-Text/dataset/annotations/captions_val2014.json'
sentences_path = 'sentences.txt'

sentences_train = parse_json_file(file_path_train)
sentences_val = parse_json_file(file_path_val)
all_sentences = sentences_train + sentences_val

sentence_array = tokenize_all_sentences(all_sentences)
write_sentence_file(sentence_array, sentences_path)




