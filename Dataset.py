from numpy import *
from Frame import *
import glob
import json
from pprint import pprint
import random
from numpy import *

class Dataset(object):
	COCO_PATH = '/home/andy/Datasets/COCO'
	COCO_TRAIN_PATH = '/train2014'
	COCO_VAL_PATH = '/val2014'
	COCO_CAPTION_TRAIN = '/annotations/captions_train2014.json'
	COCO_CAPTION_VAL = '/annotations/captions_val2014.json'


	FLICKR_PATH = '/home/andy/Datasets/Flickr'
		
	def __init__(self, numb_samples=200, perc_train=0.8, is_data_coco=True):
		self.is_data_coco = is_data_coco
		self.frames = []		
		self.numb_samples = numb_samples		
		self.perc_train=perc_train

		self.train_frames = []
		self.test_frames = []

		if is_data_coco:
			self.data_path = self.COCO_PATH
			self.load_coco_images()
			self.load_coco_captions()
		else:
			self.data_path = self.FLICKR_PATH
			self.load_flickr_images()
			self.load_flickr_captions()

		
		self.numb_frames = len(self.frames)
		self.split_data()

		del self.frames

		self.train_avg_img = None
		self.test_avg_img = None
		self.train_var_img = None
		self.test_var_img = None
		
	
	def split_data(self):
		print 'Splitting Frames:'
		numb_train_damples = self.perc_train * len(self.frames)
		
		for i in range(self.numb_frames):
			if i < numb_train_damples:
				self.train_frames.append(self.frames[i])
			else:
				self.test_frames.append(self.frames[i])

	def load_coco_images(self):
		print 'Loading Images:'
		val_path = self.COCO_PATH + self.COCO_VAL_PATH
		train_path = self.COCO_PATH + self.COCO_TRAIN_PATH

		val_images_paths = glob.glob(val_path+'/*')
		train_images_paths = glob.glob(train_path+'/*')
		
		images_paths = val_images_paths + train_images_paths
		#random.shuffle(images_paths)
		images_paths.sort()
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
		return self.numb_frames

	def get_test_frame_at(self, pos):
		return self.test_frames[pos]

	def get_train_frame_at(self, pos):
		return self.train_frames[pos]

	def get_train_frames(self):
		return self.train_frames

	def get_test_frames(self):
		return self.test_frames

	def get_numb_test_frames(self):
		return len(self.test_frames)

	def get_numb_train_frames(self):
		return len(self.train_frames)

	#Objective: 0 mean and unit variance
	def preprocess_img_train_dataset(self):
		print 'Preprocessing Images Train Set'
		self.train_avg_img = self.calc_avg_img(self.train_frames)
		self.train_var_img = self.calc_var_img(self.train_frames, self.train_avg_img)

		for frame in self.train_frames:
			frame.set_image((frame.get_image() - self.train_avg_img)/self.train_var_img)

	def preprocess_img_test_dataset(self):
		print 'Preprocessing Images Test Set'
		self.test_avg_img = self.calc_avg_img(self.test_frames)
		self.test_var_img = self.calc_var_img(self.test_frames, self.test_avg_img)

		for frame in self.test_frames:
			frame.set_image((frame.get_image() - self.train_avg_img)/self.train_var_img)
	
	def get_test_avg_img(self):
		return self.test_avg_img

	def get_train_avg_img(self):
		return self.train_avg_img

	def get_test_var_img(self):
		return self.test_var_img

	def get_train_var_img(self):
		return self.train_var_img

	def calc_avg_img(self, frames):
		avg_img = zeros(frames[0].get_shape())*1.0
		
		for frame in frames:
			avg_img += frame.get_image()

		avg_img /= len(frames)
		return avg_img

	def calc_var_img(self, frames, avg_img):
		var_img = zeros(frames[0].get_shape())*1.0
		
		for frame in frames:
			var_img += ((frame.get_image())**2)/len(frames)

		var_img -= avg_img**2
		return var_img


	
