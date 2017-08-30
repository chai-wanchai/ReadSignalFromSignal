######### Check Sampling Rate #############
from os import listdir
from os.path import isfile, join
import pyedflib
import numpy as np
from scipy.stats import kurtosis
### You must create folder name = "edf" and put file .edf in folder ###
mypath ="..\edf\edf"
FileName = [f for f in listdir(mypath)]
for numFile in np.arange(len(FileName)):
	fullpath = mypath+"\\"+FileName[numFile]
	print(numFile+1,"-----------------"+FileName[numFile]+"-----------------")
	###################  Section read EDF file ##################
	f = pyedflib.EdfReader(fullpath)
	n = f.signals_in_file
	print("signal:",n)
	signal_labels = f.getSignalLabels()

	for i in np.arange(n):
		
		sampleN =f.getNSamples()[i]
		ff = f.readSignal(i)
		d = sampleN/min(f.getNSamples())
		print(signal_labels[i])
		print(d)
	
	print("\n")	
	

	############### end section read file EDF ##############

