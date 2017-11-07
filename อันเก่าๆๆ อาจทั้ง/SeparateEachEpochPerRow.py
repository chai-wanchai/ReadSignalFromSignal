import pandas as pd
import datetime as T
import dask.dataframe as dd
import os
import numpy as np


s = T.datetime.now()
path = input("Input path of CSV file:")
filename = [f for f in os.listdir(path) if f.endswith('.csv')]
for i in range(len(filename)):
	print(i,filename[i])
select = (int)(input('Select File:'))
Fullfilename=path+"\\"+filename[select]
df = pd.read_csv(Fullfilename,dtype=object,header=1)

##############  convert to 1 dimension array #######################
def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList

#################################################################

Signal = ['subject','epoch',
          'SaO2_min', 'SaO2_max', 'SaO2_avg', 'SaO2_SD', 'SaO2_kurtosis',
          'PR_min', 'PR_max', 'PR_avg', 'PR_SD', 'PR_kurtosis',
          'position_min', 'position_max', 'position_avg', 'position_SD', 'position_kurtosis',
          'light_min', 'light_max', 'light_avg', 'light_SD', 'light_kurtosis',
          'ox_stat_min', 'ox_stat_max', 'ox_stat_avg', 'ox_stat_SD', 'ox_stat_kurtosis',
          'airflow_min', 'airflow_max', 'airflow_avg', 'airflow_SD', 'airflow_kurtosis',
          'ThorRes_min', 'ThorRes_max', 'ThorRes_avg', 'ThorRes_SD', 'ThorRes_kurtosis',
          'AbdoRes_min', 'AbdoRes_max', 'AbdoRes_avg', 'AbdoRes_SD', 'AbdoRes_kurtosis',
          'EOG(L)_min', 'EOG(L)_max', 'EOG(L)_avg', 'EOG(L)_SD', 'EOG(L)_kurtosis',
          'EOG(R)_min', 'EOG(R)_max', 'EOG(R)_avg', 'EOG(R)_SD', 'EOG(R)_kurtosis',
          'EEG(sec)_min', 'EEG(sec)_max', 'EEG(sec)_avg', 'EEG(sec)_SD', 'EEG(sec)_kurtosis',
          'EEG_min', 'EEG_max', 'EEG_avg', 'EEG_SD', 'EEG_kurtosis',
          'ECG_min', 'ECG_max', 'ECG_avg', 'ECG_SD', 'ECG_kurtosis',
          'EMG_min', 'EMG_max', 'EMG_avg', 'EMG_SD', 'EMG_kurtosis',
          'stage']
separateEachEpoch = []
tempPerEpoch =[]
for col in range(1,len(df.columns)):
    epoch = 1
    for row in range(1,len(df.values)):

        if pd.isnull(df.values[row][col]):
            break
        else:
            convertDacimal = round((float)(df.values[row][col]),10)
            tempPerEpoch.append(convertDacimal)
        if df.values[row][0]=='e'+str(epoch)+'_stage' :
            #string = str(tempPerEpoch).replace("'","").replace("[","").replace("]","")
            a = [df.values[0][col],epoch,tempPerEpoch]
            a = flatten(a)
            separateEachEpoch.append(a)
            tempPerEpoch=[]
            epoch+=1

del tempPerEpoch,a,convertDacimal


#print(separateEachEpoch)

Data = pd.DataFrame(separateEachEpoch,columns=Signal)
outfile=path+"\\SeparateEachEpochPerRow_"+filename[select]
Data.to_csv(outfile, encoding='utf-8', index=False)
print(outfile)

e = T.datetime.now()
print(e-s)