import os
import sys
import datetime
import pdb
def make_result_def(folder_path,model_name):
	now = datetime.datetime.now()
	file_date  = (str(now.date().month)+'_' + str(now.date().day) + '_' + str(now.time().hour) + '_' + str(now.time().minute))
	file_name  = model_name + '_' + file_date
	folder_path = folder_path +'/'+file_name
	#print (file_name)
	os.makedirs(folder_path)
	os.makedirs(folder_path +'/checkpoints')
	os.makedirs(folder_path +'/result')
	os.makedirs(folder_path +'/log')
	return folder_path
	


if __name__=='__main__':
	make_result_def('.','trail')
