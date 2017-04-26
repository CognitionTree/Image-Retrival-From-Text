from numpy import *
from Frame import *
import glob
import json
from pprint import pprint
import random
from numpy import *

class Dataset(object):
	COCO_PATH = '/Users/danielaflorit/Github/COCO_Dataset'
	COCO_TRAIN_PATH = '/train2014'
	COCO_VAL_PATH = '/val2014'
	COCO_CAPTION_TRAIN = '/annotations/captions_train2014.json'
	COCO_CAPTION_VAL = '/annotations/captions_val2014.json'

	def __init__(self, numb_samples=100, perc_train=0.8, numb_captions = 5):
		self.frames = []		
		self.numb_samples = numb_samples		
		self.perc_train=perc_train
		self.numb_captions = numb_captions

		self.train_frames = []
		self.test_frames = []
		
		self.data_path = self.COCO_PATH
		self.load_coco_images()
		self.load_coco_captions()
		
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
		print 'Loading Validation Captions: '
		val_caption_path = self.COCO_PATH + self.COCO_CAPTION_VAL
		print 'Loading Training Captions: '
		train_caption_path = self.COCO_PATH + self.COCO_CAPTION_TRAIN
		
		val_captions, val_captions_ids = self.read_json_file_captions(val_caption_path)
		train_captions, train_captions_ids = self.read_json_file_captions(train_caption_path)
		
		self.map_caption_to_frame(val_captions, val_captions_ids)
		self.map_caption_to_frame(train_captions, train_captions_ids)

	def map_caption_to_frame(self, captions, captions_ids):
		
		for i in range(len(self.frames)):
			frame = self.frames[i]
			frame_id = frame.get_id()
			if frame_id in captions:
				str_captions = captions[frame_id][0:self.numb_captions]
				int_captions_ids = captions_ids[frame_id][0:self.numb_captions]
				
				frame.set_caption_ids(int_captions_ids)
				frame.set_captions_text(str_captions)
			
			self.print_perc(i, len(self.frames), 5)	

	def read_json_file_captions(self, caption_path):
		with open(caption_path) as data_file:
			temp_data = json.load(data_file)
		temp_data = temp_data['annotations']
		data_caption = {}
		data_caption_ids = {}
		
		for d in temp_data:
			img_id = int(d['image_id'])
			img_caption = str(d['caption'])
			img_caption_ids = int(d['id'])
			
			if img_id in data_caption:
				data_caption[img_id].append(img_caption)
				data_caption_ids[img_id].append(img_caption_ids)
			else:
				data_caption[img_id] = [img_caption]
				data_caption_ids[img_id] = [img_caption_ids]

		return data_caption, data_caption_ids 		
		
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

	def load_pairs(self, is_training=True):
		I,T,y = None, None, None		
	
		if is_training:
			#print 'Create pairs out of the training dataset'
			I,T,y = self.create_pairs(self.train_frames)
		else:
			#print 'Create pairs out of the testing dataset'
			I,T,y = self.create_pairs(self.test_frames)
		
		return I,T,y 

	def create_pairs(self, frames, nub_neg_frames=8):
		I = []
		T = []
		y = []
		
		for frame in frames:
			id = frame.get_id()
			captions_embs = frame.get_captions_embeding()
			img = frame.get_image()

			for cap_emb in captions_embs:
				y.append(1)
				I.append(img)
				T.append(cap_emb)
			
			#For negative pairs
			neg_pairs_count = 0
			while neg_pairs_count < nub_neg_frames:
				frame2 = frames[random.randint(0,len(frames))]
				id2 = frame2.get_id()
				captions_embs2 = frame2.get_captions_embeding()
				captions_embs2 = captions_embs2[random.randint(0,len(captions_embs2))]
				img2 = frame2.get_image()

				if id != id2:
					I.append(img2)
					T.append(captions_embs2)
					y.append(0)
					neg_pairs_count+=1

		return array(I), array(T), array(y)
