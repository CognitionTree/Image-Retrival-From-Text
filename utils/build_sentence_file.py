import json
#from numpy import *
#from nltk.tokenize import word_tokenize

def parse_json_file(file_path):
	file = json.loads(open(file_path).read())
	captions = file['annotations']
	for caption in captions:
		sentence = caption['caption']
		sentence = word_tokenize(sentence)
		for word in sentence:
			word = word.lower()
			

#-----Main----
file_path = '/Users/danielaflorit/Downloads/annotations/captions_train2014.json'
parse_json_file(file_path)



