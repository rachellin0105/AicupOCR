
# change to contest form
import re
import glob,os 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_path',type=str, default="./results",
						help='path of results')
parser.add_argument('--output',type=str,default="./DB_result_AIcuptype.txt")
arg = parser.parse_args()

file_list = os.listdir(arg.results_path)
for file in sorted(file_list):
	if file.endswith(".txt"):
		path = os.path.join(arg.results_path,file)
		num = int(re.search(r'\d+',file).group())
		with open( path,'r',encoding='utf8' ) as rf:
			lines = rf.readlines()
			for line in lines:
				with open(arg.output,'a',encoding='utf8' ) as wf:
					wf.write("{},{}".format(num,line))

