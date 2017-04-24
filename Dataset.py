from numpy import *
from Frame import *
import glob
import json
from pprint import pprint

class Dataset(object):
	COCO_PATH = '/home/andy/Datasets/COCO'
	COCO_TRAIN_PATH = '/train2014'
	COCO_VAL_PATH = '/val2014'
	COCO_CAPTION_TRAIN = '/annotations/captions_train2014.json'
	COCO_CAPTION_VAL = '/annotations/captions_val2014.json'


	FLICKR_PATH = '/home/andy/Datasets/Flickr'
		
	def __init__(self, is_data_coco=True):
		self.is_data_coco = is_data_coco
		self.frames = []		

		if is_data_coco:
			self.data_path = self.COCO_PATH
			self.load_coco_images()
			self.load_coco_captions()
		else:
			self.data_path = self.FLICKR_PATH
			self.load_flickr_images()
			self.load_flickr_captions()
	
	def load_coco_images(self):
		val_path = self.COCO_PATH + self.COCO_VAL_PATH
		train_path = self.COCO_PATH + self.COCO_TRAIN_PATH

		val_images_paths = glob.glob(val_path+'/*')
		train_images_paths = glob.glob(train_path+'/*')
		
		images_paths = val_images + train_images
		
		for image_path in images_paths:
			frame = Frame(image_path)
			self.frames.append(frame)
			

	def load_coco_captions(self):
		#Set the text
		val_caption_path = self.COCO_PATH + self.COCO_CAPTION_VAL
		train_caption_path = self.COCO_PATH + self.COCO_CAPTION_TRAIN
		
		val_captions = self.read_file_captions(val_caption_path)
		train_captions = self.read_file_captions(train_caption_path)
		#Set the embeding

	def load_flickr_images(self):
		print 'Implement This'

	def load_flickr_captions(self):
		print 'Implement this' 
		
		
