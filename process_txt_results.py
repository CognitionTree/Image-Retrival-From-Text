import numpy
from nltk.tokenize import word_tokenize

file_names = ["lstm_avg50eoch.txt", "lstm_final_timestep_50epc.txt", "lstm_max50eoch.txt", "oned_50epch.txt"]

for file_name in file_names:
	file = open(file_name, 'r')

	loss = []

	for line in file:
		line = word_tokenize(line)
	
		if "loss" in line:
			val = line[line.index("loss") + 2]
			loss.append(float(val))
	
	print "++++++++++++++++++++++++++++++++++++"
	print file_name
	print loss
	print "length ", len(loss)
	file.close()