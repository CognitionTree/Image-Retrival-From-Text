import cv2
from numpy import *
import sys
from utils import *

#TODO: Decide to save or get rid of: path and original caption
class Frame:
	
	#is_data_coco = True means use coco dataset otherwise use Flickr
	def __init__(self, frame_path, new_rows=640/10, new_cols=426/10, is_data_coco=True, sentence_size=None, captions_text=None, load_emb = True):
		self.frame_path = frame_path
		self.is_data_coco = is_data_coco
		self.load_emb = load_emb #If True, load caption embeddings, otherwise, compute them
		
		self.img = None
		self.id = None
		self.captions_text = captions_text
		self.captions_embs = []
		self.caption_ids = []

		self.parse_path_data(frame_path)
		self.resize_frame(new_rows, new_cols)

	def parse_path_data(self, frame_path):
		self.img = cv2.imread(frame_path)
        
		#isolating frame name from the rest of the path
		splited_path = frame_path.split('/')
		frame_name = splited_path[-1]
		frame_name = frame_name.split('.')[0]

		if self.is_data_coco:
			self.id = self.parse_coco_frame_name(frame_name)
		else:
			self.id = self.parse_flickr_frame_name(frame_name)

	def parse_coco_frame_name(self, frame_name):
		frame_name = frame_name.split('_')[-1]
		return int(frame_name)
		
	def parse_flickr_frame_name(self, frame_name):
		return int(frame_name)
	
	def get_path(self):
		return self.frame_path

	def get_id(self):
		return self.id

	def get_shape(self):
		return self.img.shape

	def get_rows(self):
		rows, cols, chans = self.img.shape
		return rows

	def get_cols(self):
		rows, cols, chans = self.img.shape
		return cols

	def get_channels(self):
		rows, cols, chans = self.img.shape
		return chans

	def get_image(self):
		return self.img

	def set_image(self, img):
		self.img = img

	def get_caption_text(self):
		return self.captions_text

	def set_captions_text(self, captions_text, multi_word=True):
		#TODO: create camption_emb which ios a numpy array
		self.captions_text = captions_text
		if self.load_emb == True:
			self.read_captions_embeding(multi_word)
		else:
			self.compute_captions_embeding(multi_word)
		
		
	#def bag_of_words_embeding():

	def get_captions_embeding(self):
		return self.captions_embs

	#It assumes that set_captions_text was already called
	def compute_captions_embeding(self, multi_word):
		#TODO: Implement this
		print "I am computing the embeddings"
		print "the embedding option is: ", multi_word
		
		for sentence in self.captions_text:
			full_emb = get_sentence_encoding(sentence)
			
			if multi_word:
				self.captions_embs.append(full_emb)
				print "---------"
				print full_emb
	
			else:
				emb = time_sum(full_emb)
				self.captions_embs.append(emb)
				print "---------"
				print emb
	
	def read_captions_embeding(self, multi_word):
		print "I am reading the embeddings"
		print "the embedding option is: ", multi_word
		
		for id in self.caption_ids:
			file_name = str(self.id) + "_" + str(id) + ".npy"
			full_emb = load(file_name)
			
			if multi_word:
				self.captions_embs.append(full_emb)
			else:
				emb = time_sum(full_emb)
				self.captions_embs.append(emb)
				
	def get_caption_ids(self):
		return self.caption_ids
	
	def set_caption_ids(self, caption_ids):
		self.caption_ids = caption_ids
	
	def resize_frame(self, new_rows, new_cols):
		#rows, cols, chans = self.img.shape
		self.img = cv2.resize(self.img, (new_cols, new_rows))

	def show(self):
		print self.captions_text
		cv2.imshow(str(self.id),self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
		
