from numpy import *
from Frame import *
import glob
import json
from pprint import pprint
import random

class Dataset(object):
	COCO_PATH = '/home/andy/Datasets/COCO'
	COCO_TRAIN_PATH = '/train2014'
	COCO_VAL_PATH = '/val2014'
	COCO_CAPTION_TRAIN = '/annotations/captions_train2014.json'
	COCO_CAPTION_VAL = '/annotations/captions_val2014.json'


	FLICKR_PATH = '/home/andy/Datasets/Flickr'
		
	def __init__(self, numb_samples=80000,is_data_coco=True):
		self.is_data_coco = is_data_coco
		self.frames = []		
		self.numb_samples = numb_samples		

		if is_data_coco:
			self.data_path = self.COCO_PATH
			self.load_coco_images()
			self.load_coco_captions()
		else:
			self.data_path = self.FLICKR_PATH
			self.load_flickr_images()
			self.load_flickr_captions()
	
	def load_coco_images(self):
		print 'Loading Images:'
		val_path = self.COCO_PATH + self.COCO_VAL_PATH
		train_path = self.COCO_PATH + self.COCO_TRAIN_PATH

		val_images_paths = glob.glob(val_path+'/*')
		train_images_paths = glob.glob(train_path+'/*')
		
		images_paths = val_images_paths + train_images_paths
		random.shuffle(images_paths)
		images_paths = images_paths[0:self.numb_samples]
				

		for i in range(len(images_paths)):
			frame = Frame(images_paths[i])
			self.frames.append(frame)
			self.print_perc(i, len(images_paths), 100)
			
	def print_perc(self, part, total, step):
		if part % step == 0:
			print str(100.0*float(part)/float(total)) + '%'	

	def load_coco_captions(self):
		#Set the text
		print 'Loading Captions: '
		val_caption_path = self.COCO_PATH + self.COCO_CAPTION_VAL
		train_caption_path = self.COCO_PATH + self.COCO_CAPTION_TRAIN
		
		val_captions = self.read_json_file_captions(val_caption_path)
		train_captions = self.read_json_file_captions(train_caption_path)
		
		self.map_caption_to_frame(val_captions)
		self.map_caption_to_frame(train_captions)

	def map_caption_to_frame(self, captions):
		#print captions.keys()
		for frame in self.frames:
			if frame.get_id() in captions: 
				str_captions = captions[frame.get_id()]
				frame.set_captions_text(str_captions)
				

	def read_json_file_captions(self, caption_path):
		with open(caption_path) as data_file:
			temp_data = json.load(data_file)
		temp_data = temp_data['annotations']
		data = {}
		for d in temp_data:
			img_id = int(d['image_id'])
			img_caption = str(d['caption'])
			
			if img_id in data:
				data[img_id].append(img_caption)
			else:
				data[img_id] = [img_caption]

		return data 		
	
	def load_flickr_images(self):
		print 'Implement This'

	def load_flickr_captions(self):
		print 'Implement this' 
		
	def __len__(self):
		return len(self.frames)

	def get_frame_at(self, pos):
		return self.frames[pos]
