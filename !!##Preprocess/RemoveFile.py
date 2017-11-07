######### Check Sampling Rate #############
from os import listdir
from os.path import isfile, join
import pyedflib
import numpy as np
from scipy.stats import kurtosis
import os
#import shutil

### You must create folder name = "edf" and put file .edf in folder ###
mypath =input("Enter Directory of EDF File: ")
move = input("Enter directory to move file: ")+"\\"
FileName = [f for f in listdir(mypath) if f.endswith(".edf")]
print(FileName)
s125=0
s128=0
for numFile in np.arange(len(FileName)):
	fullpath = mypath+"\\"+FileName[numFile]
	
	print(numFile+1,"-----------------"+FileName[numFile]+"-----------------")
	###################  Section read EDF file ##################
	f = pyedflib.EdfReader(fullpath)
	n = f.signals_in_file
	sampleN = f.getSampleFrequency(2)
	print(sampleN)
	f._close()

	if (sampleN==125):
		s125=s125+1
		
	elif (sampleN==128):
		s128=s128+1
		#print("128")
print("sample 125:",s125)
print("sample 128:",s128)
	# 	os.rename(fullpath,move+FileName[numFile])
	# 	print("move complete"+FileName[numFile])
	# else:
	# 	if not os.path.exists(move+'otherFrequency'):
	# 		os.makedirs(move+'otherFrequency')
	# 		os.rename(fullpath,move+'otherFrequency'+'\\'+FileName[numFile])
	# 	else:
	# 		os.rename(fullpath,move+'otherFrequency'+'\\'+FileName[numFile])
	# #shutil.move("..\\edf\\edf\\test.txt", "..\\edf\\EDF_8Hz\\")
	

	############### end section read file EDF ##############

