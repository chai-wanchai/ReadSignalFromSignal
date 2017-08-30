from os import listdir
from os.path import isfile, join
import os.path
import pyedflib
import numpy as np
from scipy.stats import kurtosis
import csv

def chunks(ls, n):
    return [ls[i:i+n] for i in range(0, len(ls), n)]

###################  Section read EDF file ##################
def ReadEDFFile (Path):
	f = pyedflib.EdfReader(Path)
	n = f.signals_in_file
	#print("signal:",n)
	signal_labels = f.getSignalLabels()
	#print("signal:",signal_labels[0])
	samplingRate =f.getNSamples()[0]/min(f.getNSamples())
	######## Frequency 1 Hz ##########
	SaO2=np.array(chunks(f.readSignal(0),1))
	PR=f.readSignal(1)
	Position=chunks(f.readSignal(11),1)
	light=chunks(f.readSignal(12),1)
	ox_stat=chunks(f.readSignal(13),1)
	######## Frequency 10 Hz ###########
	airflow=chunks(f.readSignal(8),10)
	ThorRes=chunks(f.readSignal(9),10)
	AbdoRes=chunks(f.readSignal(10),10)
	########  Frequency 50 Hz ##########
	EOG_L =chunks(f.readSignal(5),50)
	EOG_R =chunks(f.readSignal(6),50)
	######### Frequency 125 Hz ########
	EEG_sec=chunks(f.readSignal(2),125)
	EEG=chunks(f.readSignal(7),125)
	EMG=chunks(f.readSignal(4),125)
	######## Frequency 250 Hz #########
	ECG=chunks(f.readSignal(3),250)
	##################################
	minN = min(f.getNSamples())
	AllSignal = np.array([])
	for I in range(1):
		AllSignal = np.array(SaO2[I])+np.array(PR[I])#+Position[I]+light[I]+ox_stat[I]+airflow[I]+ThorRes[I]+AbdoRes[I]+min(EOG_L[I])+max(EOG_L[I])+np.mean(EOG_L[I])+np.std(EOG_L[I])+kurtosis(EOG_L[I])+min(EOG_R[I])+max(EOG_R[I])+np.mean(EOG_R[I])+np.std(EOG_R[I])+kurtosis(EOG_R[I])+min(EEG_sec[I])+max(EEG_sec[I])+np.mean(EEG_sec[I])+np.std(EEG_sec[I])+kurtosis(EEG_sec[I])+min(EEG[I])+max(EEG[I])+np.mean(EEG[I])+np.std(EEG[I])+kurtosis(EEG[I])+min(ECG[I])+max(ECG[I])+np.mean(ECG[I])+np.std(ECG[I])+kurtosis(ECG[I])+min(EMG[I])+max(EMG[I])+np.mean(EMG[I])+np.std(EMG[I])+kurtosis(EMG[I])
		print('sa',np.array(SaO2))
		#w=PR.reshape(minN,2)
		print('PR',PR)
		print(AllSignal)
	
'''
	for i in np.arange(n):
		print("signal:",i,signal_labels[i])
		samplingRate =f.getNSamples()[i]/min(f.getNSamples())
		#print(sampleN)
		ff = np.array(f.readSignal(i))
		
		if (samplingRate  == 1) : 
			if (i==0):SaO2 =chunks(ff, 1)
			if (i==1):PR =chunks(ff, 1)
			if (i==11):Position = chunks(ff, 1)
			if (i==12):light=chunks(ff, 1)
			if (i==13):ox_stat =chunks(ff, 1)
				#print(x_stat[500])
			#print("d",PR[1])
		if (samplingRate  == 10) : 
			if (i==8): airflow = chunks(ff, 10)
			if (i==9): ThorRes = chunks(ff, 10)
			if (i==10): AbdoRes = chunks(ff, 10)
		
		if (samplingRate  == 50) : 
			if (i==5):EOG_L = chunks(ff, 50)
			if (i==6):EOG_R= chunks(ff, 50)
		#print("EOG",EOG_L)
		if (samplingRate  == 125) : 
			if (i==2):EEG_sec = chunks(ff, 125)
			if (i==7):EEG =chunks(ff, 125)
			if (i==4):EMG =chunks(ff, 125)
					
			
		if (samplingRate  == 250) :
			if (i==3):ECG = chunks(ff, 250)
		
'''
	#print('PR1:',PR[1])
	#minN = min(f.getNSamples())
	#for I in range(minN-1):
		#AllSignal = SaO2[I]+PR[I]+Position[I]+light[I]+ox_stat[I]+airflow[I]+ThorRes[I]+AbdoRes[I]+min(EOG_L[I])+max(EOG_L[I])+np.mean(EOG_L[I])+np.std(EOG_L[I])+kurtosis(EOG_L[I])+min(EOG_R[I])+max(EOG_R[I])+np.mean(EOG_R[I])+np.std(EOG_R[I])+kurtosis(EOG_R[I])+min(EEG_sec[I])+max(EEG_sec[I])+np.mean(EEG_sec[I])+np.std(EEG_sec[I])+kurtosis(EEG_sec[I])+min(EEG[I])+max(EEG[I])+np.mean(EEG[I])+np.std(EEG[I])+kurtosis(EEG[I])+min(ECG[I])+max(ECG[I])+np.mean(ECG[I])+np.std(ECG[I])+kurtosis(ECG[I])+min(EMG[I])+max(EMG[I])+np.mean(EMG[I])+np.std(EMG[I])+kurtosis(EMG[I])
		
	#print(AllSignal)
		
		
	#print("\n")
	#return
############### end section read file EDF ##############


########################  Write CSV File #############################################
def WriteHeaderCSV (listData):
	if os.path.isfile("SignalEpoch30s.csv"):
		print ("Have File")
		with open('SignalEpoch30s.csv', 'a', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow(listData)
		
	else:
		###Create header og CSV as e1_EEG(sec) mean epoch 1 of signal EEG
		print ("don't have file")
		s =['subject']
		Signal = ['SaO2','PR','position','light','ox_stat','airflow','ThorRes','AbdoRes',
			'EOG(L)_min','EOG(L)_max','EOG(L)_avg','EOG(L)_SD','EOG(L)_kurtosis',
			'EOG(R)_min','EOG(R)_max','EOG(R)_avg','EOG(R)_SD','EOG(R)_kurtosis',
			'EEG(sec)_min','EEG(sec)_max','EEG(sec)_avg','EEG(sec)_SD','EEG(sec)_kurtosis',
			'EEG_min','EEG_max','EEG_avg','EEG_SD','EEG_kurtosis',
			'ECG_min','ECG_max','ECG_avg','ECG_SD','ECG_kurtosis',
			'EMG_min','EMG_max','EMG_avg','EMG_SD','EMG_kurtosis']
		for n in range(1440):
			for i in Signal:
				s.append('e'+str(n+1)+'_'+i)
		
		with open('SignalEpoch30s.csv', 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow(s)
			#spamwriter.close()
########################################################################################	




#########################  Main #######################################################
if __name__ == '__main__':
	### You must create folder name = "edf" and put file .edf in folder ###
	mypath ="..\edf\edf"
	FileName = [f for f in listdir(mypath)]
	for numFile in FileName:
		fullpath = mypath+"\\"+numFile
		print("-----------------"+numFile+"-----------------")
		ReadEDFFile(fullpath)
		WriteHeaderCSV(FileName)

#########################  End Main  ##################################################