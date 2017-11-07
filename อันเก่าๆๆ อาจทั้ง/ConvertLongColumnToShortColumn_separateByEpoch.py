import csv
import pandas as pd
import datetime as Time
import os
a = Time.datetime.now()
print("This script is convert Data in CSV to column separate by epoch.")
path =input("Enter path of CSV file :")
infile = [f for f in os.listdir(path) if f.endswith(".csv")]
for i in range(len(infile)):
    print(i,infile[i])
NumberOfFile = (int)(input("Enter number File :"))
print("open file: ",path+"\\"+infile[NumberOfFile])
fullPathOpen =path+"\\"+infile[NumberOfFile]
outPath =input("Enter path to save it :")
outfile = input("Enter file output name : ")+".csv"
fullPathOut = outPath+'\\'+outfile

with open(fullPathOpen, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    data =[]
    countRow=0
    for read in spamreader:
        data.append(read)

    #print(len(data))
    #print(len(data[1]))
    temp=[]
    #person=[]
    perEpoch=[]
    for row in range(1,len(data)):
        epoch=1
        for col in range(1,len(data[row])):
            if col%71==0:
                temp.append(round((float)(data[row][col]),10))
                person=[data[row][0],epoch]+temp
                perEpoch.append(person)
                person=[]
                temp=[]
                epoch+=1

            else:temp.append(round((float)(data[row][col]),10))

    Signal = ['subject', 'epoch',
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
    #print(len(perEpoch))
    Frame = pd.DataFrame(perEpoch,columns=Signal)
    Frame.to_csv(fullPathOut, encoding='utf-8',index=False)
print(fullPathOut)

