#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from DB.experiment import Structure, Experiment
from DB.concern.config import Configurable, Config
import math

class DB_detection:
	def __init__(self, args):
		args = vars(args)
		args = {k: v for k, v in args.items() if v is not None}
		experiment,experiment_args= self.init_config(args)

		self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
		self.experiment = experiment
		experiment.load('evaluation', **experiment_args)
		self.args = args
		model_saver = experiment.train.model_saver
		self.structure = experiment.structure
		self.model_path = self.args['resume']
		self.args['image_short_side'] = 1024

		self.init_torch_tensor()
		model = self.init_model()
		self.resume(model, self.model_path)
		model.eval()
		self.model = model

	def init_config(self,args):
		conf = Config()
		experiment_args = conf.compile(conf.load(args['config_file_DB']))['Experiment']
		experiment_args.update(cmd=args)
		experiment = Configurable.construct_class_from_config(experiment_args)
		return experiment,experiment_args

	def init_torch_tensor(self):
		# Use gpu or not
		torch.set_default_tensor_type('torch.FloatTensor')
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
		else:
			self.device = torch.device('cpu')

	def init_model(self):
		model = self.structure.builder.build(self.device)
		return model

	def resume(self, model, path):
		if not os.path.exists(path):
			print("Checkpoint not found: " + path)
			return
		print("Resuming from " + path)
		states = torch.load(
			path, map_location=self.device)
		model.load_state_dict(states, strict=False)
		print("Resumed from " + path)

	def resize_image(self, img):
		height, width, _ = img.shape
		if height < width:
			new_height = self.args['image_short_side']
			new_width = int(math.ceil(new_height / height * width / 32) * 32)
		else:
			new_width = self.args['image_short_side']
			new_height = int(math.ceil(new_width / width * height / 32) * 32)
		resized_img = cv2.resize(img, (new_width, new_height))
		return resized_img
		
	def load_image(self, image_path):
		img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
		original_shape = img.shape[:2]
		img = self.resize_image(img)
		img -= self.RGB_MEAN
		img /= 255.
		img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
		return img, original_shape
		
	def format_output(self, batch, output):
		batch_boxes, batch_scores = output
		result = []
		for index in range(batch['image'].size(0)):
			original_shape = batch['shape'][index]
			filename = batch['filename'][index]
			result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
			result_file_path = os.path.join(self.args['result_dir'], result_file_name)
			boxes = batch_boxes[index]
			scores = batch_scores[index]

			for i in range(boxes.shape[0]):
				score = scores[i]
				if score < self.args['box_thresh']:
					continue
				box = boxes[i,:,:].reshape(-1).tolist()
				result.append(",".join([str(int(x)) for x in box]))
				
		return result
					
		
	def inference(self, image_path, visualize=False):
		batch = dict()
		batch['filename'] = [image_path]
		img, original_shape = self.load_image(image_path)
		batch['shape'] = [original_shape]
		with torch.no_grad():
			batch['image'] = img
			pred = self.model.forward(batch, training=False)
			output = self.structure.representer.represent(batch, pred, is_output_polygon=False) 
			if not os.path.isdir(self.args['result_dir']):
				os.mkdir(self.args['result_dir'])
			return self.format_output(batch, output)

				
