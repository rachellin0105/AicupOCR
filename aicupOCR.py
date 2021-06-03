#!python3
import argparse
import os
from tqdm import tqdm
from DB.demo import DB_detection
from CRNN.demo import CRNN_recognition
import re
from draw_result import drawResult

def main():

	parser = argparse.ArgumentParser(description='AICUP')
	

	# general argument
	parser.add_argument('--image_dir_path', type=str,  default='./image_all',help='image path')
	parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
	parser.add_argument('--output_file_name', type=str,help='')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--virtualize',type=bool,default = False, help='virtualize the result')
	# for aicap
	parser.add_argument('--aicup2_file', type=str, default='',help=' GT of boxes ; aicup2 case if given else normal case') #EX: ./AdvancedComp_public_test/easy/coordinates_easy.txt
	# DB argument
	parser.add_argument('--config_file_DB', type=str,default='./config_file/AICUP_resnet50_deform_thre.yaml')
	parser.add_argument('--resume', type=str, default='model/DB_aicup1_final',help='Resume from checkpoint')
	parser.add_argument('--box_thresh', type=float, default=0.5,
		help='The threshold to replace it in the representers')
	parser.add_argument('--visualize', action='store_true',
					help='visualize maps in tensorboard')
	# CRNN argument
	parser.add_argument('--saved_model', default= './model/without_dataaug_ch.pth',help="path to saved_model to evaluation")
	parser.add_argument('--batch_max_length', type=int, default=30, help='maximum-label-length')
	parser.add_argument('--rgb', action='store_true', help='use rgb input')
	parser.add_argument('--dic_file', type=str, default='./CRNN/dic_ch.txt', help='use rgb input')
	parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
	parser.add_argument('--threshold_crnn', type=float, default=0.1, help='threshold for crnn')
	args = parser.parse_args()



	'''
	image_info:
		dictionary img_boxes
			key : img_{num}.jpg
			value(boxs_info_per_image) : list[(box,label)] box=[x,y,x,y,x,y,x,y]
	'''
	image_info = {}
	'''
	detectionResult_dic:
		key: img_{num}.jpg
		value: list of boxes
			EX: ['745,294,841,294,841,352,745,352', '37,195,346,186,383,1335,74,1345']
	'''
	detectionResult_dic = {}

	'''
	create detector and recognizer
	'''
	db_detection =  DB_detection(args)
	crnn_recognition = CRNN_recognition(args)

	'''
	create a result dir if not exit
	only output one file(txt)
	'''
	if not os.path.exists(args.result_dir):
		os.mkdir(args.result_dir)

	output_file_path = os.path.join(args.result_dir,args.output_file_name)

	writeLog(args)
	'''
	Do detection
	'''
	if args.aicup2_file:
		detectionResult_dic = loadBoxes(args.image_dir_path,args.aicup2_file)
	else:
		img_list = os.listdir(args.image_dir_path)

		for img_name in sorted(img_list):
			image_path = os.path.join(args.image_dir_path,img_name)
			boxes = db_detection.inference(image_path, True)
			detectionResult_dic[img_name] = boxes

	'''
	Do recognition
	'''
	for img_name,boxes in tqdm(detectionResult_dic.items()):
		image_path = os.path.join(args.image_dir_path,img_name)
		if boxes:
			labal_list,confidence_scores = crnn_recognition.recognize(image_path,boxes)

			boxs_info = [(box,label)for box,label in zip(boxes,labal_list)]
			'''
			output results
			'''

			outputResult(img_name,boxs_info,confidence_scores,output_file_path,args.threshold_crnn)
			if args.visualize:
				drawResult(img_name,boxs_info,args)

def writeLog(args):
	logPath = os.path.join(args.result_dir,'args_log.txt') 
	with open(logPath,'w',encoding="utf8") as log:
			log.write('image_dir_path = {}\nthreshold_crnn = {}\naicup2_file={}\n'.format(args.image_dir_path,args.threshold_crnn,args.aicup2_file))



def outputResult(img_name,info,confidence_scores,output_file_path,threshold_score):
	assert len(info) == len(confidence_scores),(info,confidence_scores)

	for (box,label), score in zip( info,confidence_scores):
		with open(output_file_path,'a', encoding = 'utf8' ) as f:
			if label == '' or score < threshold_score:
				label = '###'
			f.write("{},{},{}\n".format(img_name.split('.')[0],box,label))


def loadBoxes(img_dir_path,file_txt):
	'''
	return:
		dictionary:
		key: img_{num}.jpg
		value: list of boxes
			EX: ['745,294,841,294,841,352,745,352', '37,195,346,186,383,1335,74,1345']
	'''
	img_boxes = {}

	with open(file_txt,'r',encoding='utf8') as f:
		lines = f.readlines()

	for index,line  in enumerate(lines):
		line = line.replace('\n','').split(',')
		img_name = '{}.jpg'.format(line[0])
		img_path = os.path.join(img_dir_path,img_name)

		if len(line[1:]) != 8 or 'img_' not in line[0] :
			print("number of points has somthing wrong in line {}.".format(index+1))
			continue

		boxes = ','.join(line[1:])
		if img_name in img_boxes:
			img_boxes[img_name].append(boxes)
		else: #add new img infomation
			
			img_boxes[img_name] = [boxes]


	return img_boxes


if __name__ == '__main__':
	main()
