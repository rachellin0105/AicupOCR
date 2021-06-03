import argparse
from tqdm import tqdm
import re
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

def loadImgInfo(txt_path):
	'''
	return:
		dictionary img_boxes
			key : img_{num}.jpg
			value : list[(box,label)]		
	'''

	img_boxes = {}

	with open(txt_path,'r',encoding='utf8') as f:
		lines = f.readlines()

	for index,line  in enumerate(lines):
		line = line.split(',')
		if len(line[1:-1]) != 8 or 'img_' not in line[0] :
			print(txt_path,line)
			print("number of points has somthing wrong in line {}.".format(index+1))
			continue

		img_name =  line[0]+'.jpg'
		box = line[1:-1]
		label = line[-1]
		info = (box,label)
		if img_name in img_boxes:
				img_boxes[img_name].append(info)
		else: # add new img infomation	
			img_boxes[img_name] = [info]

	return img_boxes

	
def drawResult(img_name,info,opt):
	'''
	draw on single image
	'''
	word_size = 20
	img_path = os.path.join(opt.image_dir_path,img_name)
	out_imgDir_path = os.path.join(opt.result_dir,'result_imgs')

	if not os.path.exists(out_imgDir_path):
		os.mkdir(out_imgDir_path)
		
	out_path = os.path.join(out_imgDir_path,img_name)
	img = cv2.imread(img_path)
	font = ImageFont.truetype('./BHEI00EU.TTF', word_size)

	for box, label in info :
		if isinstance( box, str ):
			box = box.split(',')
		box = [int(b) for b in box]
		box = np.array(box).reshape(-1, 2)
		# draw box
		for i in range(len(box)-1) :
			cv2.line(img,tuple(box[i]),tuple(box[i+1]),(0,255,0),2, cv2.LINE_AA)
		cv2.line(img,tuple(box[-1]),tuple(box[0]),(0,255,0),2, cv2.LINE_AA)

		# draw white aera & draw label

		if box[2][0]-box[0][0] < box[2][1] - box[0][1]:#vertial
			cv2.rectangle(img, (box[0][0]-word_size,box[0][1]), (box[0][0],box[0][1]+len(label)*word_size), (255, 255, 255), -1)
			imgPil = Image.fromarray(img)
			draw = ImageDraw.Draw(imgPil)
			for index,word in enumerate(label):
				draw.text((box[0][0]-word_size,box[0][1]+index*word_size),  word, font = font, fill = (0, 0, 0))

		else:
			cv2.rectangle(img, (box[0][0],box[0][1]-word_size), (box[0][0]+len(label)*word_size,box[0][1]), (255, 255, 255), -1)
			imgPil = Image.fromarray(img)
			draw = ImageDraw.Draw(imgPil)
			draw.text((box[0][0],box[0][1]-word_size),  label, font = font, fill = (0, 0, 0))

		img = np.array(imgPil)
	#print(out_path)
	cv2.imwrite(out_path,img)

		



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='draw_reult')
	parser.add_argument('result_txt', type=str)
	parser.add_argument('image_dir_path', type=str)
	parser.add_argument('result_dir', type=str)
	args = parser.parse_args()

	if not os.path.exists(args.result_dir):
		os.mkdir(args.result_dir)

	img_info = loadImgInfo(args.result_txt)

	for img_name , info in tqdm(img_info.items()):
		drawResult(img_name,info,args)






