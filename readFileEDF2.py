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
		print("signal:",signal_labels[i])
		sampleN =f.getNSamples()[i]
		print(sampleN)
		ff = f.readSignal(i)
		print(ff)
		print("max:",max(ff))
		print("min:",min(ff))
		print("avg:",np.mean(ff))
		print("s.d.:",np.std(ff))
		print("kurtosis:",kurtosis(ff))
		print("sampling rate:",sampleN/min(f.getNSamples()))
		print("\n")	
	print("max record" ,max(f.getNSamples()))
	print("min record" ,min(f.getNSamples()))
	print("\n")

	############### end section read file EDF ##############

	
