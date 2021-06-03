import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from CRNN.utils import CTCLabelConverter, AttnLabelConverter
from CRNN.dataset import CropImageDataset, AlignCollate
from CRNN.model import Model


class CRNN_recognition:
	def __init__(self, opt):
		opt.FeatureExtraction = "ResNet"
		opt.SequenceModeling = "BiLSTM" 
		opt.Prediction = "CTC" 
		opt.Transformation = "TPS" 
		opt.num_fiducial = 20
		opt.output_channel = 512
		opt.input_channel = 1
		opt.hidden_size = 256
		opt.imgH = 32
		opt.imgW = 320
		try:
			with open(opt.dic_file,'r',encoding="utf8") as f :
				opt.character = [char.replace("\n","") for char in f.readlines()]
			
		except:
			assert False , "something wrong on getting dic"

		cudnn.benchmark = True
		cudnn.deterministic = True
		opt.num_gpu = torch.cuda.device_count()

		self.opt = opt
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		""" model configuration """
		if 'CTC' in opt.Prediction:
			self.converter = CTCLabelConverter(opt.character)
		else:
			self.converter = AttnLabelConverter(opt.character)
		opt.num_class = len(self.converter.character)

		if opt.rgb:
			opt.input_channel = 3
		model = Model(opt)
		model = torch.nn.DataParallel(model).to(self.device)

		# load model
		print('loading pretrained model from %s' % opt.saved_model)
		model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))
		model.eval()
		self.model = model

	def load_data(self,image_path,boxes):
		opt = self.opt
		# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
		AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
		demo_data = CropImageDataset(image_path=image_path,boxes=boxes, opt=opt)  # use RawDataset

		demo_loader = torch.utils.data.DataLoader(
			demo_data, batch_size=len(boxes),
			shuffle=False,
			num_workers=int(opt.workers),
			collate_fn=AlignCollate_demo, pin_memory=True)

		return demo_loader

	def recognize(self,image_path,boxes):
		opt = self.opt
		demo_loader = self.load_data(image_path,boxes)
		# predict
		confidence_score = []

		with torch.no_grad():
			for image_tensors, image_path_list in demo_loader:
				batch_size = image_tensors.size(0)
				image = image_tensors.to(self.device)
				# For max length prediction
				length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
				text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)

				if 'CTC' in opt.Prediction:
					preds = self.model(image, text_for_pred)

					# Select max probabilty (greedy decoding) then decode index to character
					preds_size = torch.IntTensor([preds.size(1)] * batch_size)
					_, preds_index = preds.max(2)
					# preds_index = preds_index.view(-1)
					preds_str = self.converter.decode(preds_index, preds_size)

				else:
					preds = self.model(image, text_for_pred, is_train=False)

					# select max probabilty (greedy decoding) then decode index to character
					_, preds_index = preds.max(2)
					preds_str = self.converter.decode(preds_index, length_for_pred)

				preds_prob = F.softmax(preds, dim=2)
				preds_max_prob, _ = preds_prob.max(dim=2)
				for probs in preds_max_prob :
					# calculate confidence score (= multiply of pred_max_prob)
					score = round(probs.cumprod(dim=0)[-1].item(), 4)
					confidence_score.append(score)

		
		return preds_str,confidence_score
				# log = open(f'./log_demo_result.txt', 'a',encoding="utf8")
				# dashed_line = '-' * 80
				# head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
				
				# print(f'{dashed_line}\n{head}\n{dashed_line}')
				# log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

				# preds_prob = F.softmax(preds, dim=2)
				# preds_max_prob, _ = preds_prob.max(dim=2)
				# for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
				# 	# calculate confidence score (= multiply of pred_max_prob)
				# 	confidence_score = pred_max_prob.cumprod(dim=0)[-1]
				# 	print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
				# 	log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

				# log.close()
