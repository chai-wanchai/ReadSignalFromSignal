import csv
########### Create header og CSV as e1_EEG(sec) mean epoch 1 of signal EEG
s =['subject']
Signal = ['SaO2','PR','position','light','ox_stat','airflow','ThorRes','AbdoRes',
			'EGO(L)_max','EGO(L)_min','EGO(L)_avg','EGO(L)_SD','EGO(L)_kurtosis',
			'EOG(R)_max','EOG(R)_min','EOG(R)_avg','EOG(R)_SD','EOG(R)_kurtosis',
			'EEG(sec)_max','EEG(sec)_min','EEG(sec)_avg','EEG(sec)_SD','EEG(sec)_kurtosis',
			'EEG_max','EEG_min','EEG_avg','EEG_SD','EEG_kurtosis',
			'ECG_max','ECG_min','ECG_avg','ECG_SD','ECG_kurtosis',
			'EMG_max','EMG_min','EMG_avg','EMG_SD','EMG_kurtosis']
for n in range(1440):
	for i in Signal:
		s.append('e'+str(n+1)+'_'+i)

################################	
with open('SignalEpoch30s.csv', 'w', newline='') as csvfile:
    
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(s)
	print("complete")
    #spamwriter.writerow()
