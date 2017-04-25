import cv2
from numpy import *
import sys
sys.path.insert(0, 'utils')
from utils import *

#TODO: Decide to save or get rid of: path and original caption
class Frame:
	
	#is_data_coco = True means use coco dataset otherwise use Flickr
	def __init__(self, frame_path, new_rows=640/6, new_cols=426/6, is_data_coco=True, sentence_size=None, captions_text=None):
		self.frame_path = frame_path
		self.is_data_coco = is_data_coco

		self.img = None
		self.id = None
		self.captions_text = captions_text
		self.captions_embs = []

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

	def set_captions_text(self, captions_text):
		self.captions_text = captions_text
		self.set_captions_embseding()
		#TODO: create camption_emb which ios a numpy array

	#def bag_of_words_embeding():

	def get_captions_embeding(self):
		return self.captions_embs

	#It assumes that set_captions_text was already called
	def set_captions_embeding(self, multi_word=True):
		#TODO: Implement this
		if multi_word:
			for sentence in self.captions_text:
				emb = get_sentence_encoding(sentence)	
				self.captions_embs.append(emb)
		 
		


	def resize_frame(self, new_rows, new_cols):
		#rows, cols, chans = self.img.shape
		self.img = cv2.resize(self.img, (new_cols, new_rows))

	def show(self):
		print self.captions_text
		cv2.imshow(str(self.id),self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
		
