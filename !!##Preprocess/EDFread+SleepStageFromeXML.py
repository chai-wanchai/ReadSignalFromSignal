from os import listdir
from os.path import isfile, join
import os.path
import pyedflib
import numpy as np
from scipy.stats import kurtosis
import csv
import xml.dom.minidom
import datetime
###########  Global Variable ##############
epoch = 10  # Config Epoch as 5s 10s 15s 30s #
PathOutputFileName = ""
XMLPath = ""

###########################################

def chunks(ls, n):
    return np.asarray([np.asarray(ls[i:i + n]) for i in range(0, len(ls), n)])


###################  Section read EDF file ##################
def ReadEDFFile(Path, FileName):
    f = pyedflib.EdfReader(Path)
    minN = min(f.getNSamples())
    ######## Frequency 1 Hz ##########
    SaO2 = chunks(f.readSignal(0), (int)(f.getSampleFrequencies()[0] * epoch))
    PR = chunks(f.readSignal(1), (int)(f.getSampleFrequencies()[1] * epoch))
    Position = chunks(f.readSignal(11), (int)(f.getSampleFrequencies()[11] * epoch))
    light = chunks(f.readSignal(12), (int)(f.getSampleFrequencies()[12] * epoch))
    ox_stat = chunks(f.readSignal(13), (int)(f.getSampleFrequencies()[13] * epoch))
    ######## Frequency 10 Hz ###########
    airflow = chunks(f.readSignal(8), (int)(f.getSampleFrequencies()[8] * epoch))
    ThorRes = chunks(f.readSignal(9), (int)(f.getSampleFrequencies()[9] * epoch))
    AbdoRes = chunks(f.readSignal(10), (int)(f.getSampleFrequencies()[10] * epoch))
    ########  Frequency 50 Hz ##########
    EOG_L = chunks(f.readSignal(5), (int)(f.getSampleFrequencies()[5] * epoch))
    EOG_R = chunks(f.readSignal(6), (int)(f.getSampleFrequencies()[6] * epoch))
    ######### Frequency 125 Hz ########
    EEG_sec = chunks(f.readSignal(2), (int)(f.getSampleFrequencies()[2] * epoch))
    EEG = chunks(f.readSignal(7), (int)(f.getSampleFrequencies()[7] * epoch))
    EMG = chunks(f.readSignal(4), (int)(f.getSampleFrequencies()[4]* epoch))
    ######## Frequency 250 Hz #########
    ECG = chunks(f.readSignal(3), (int)(f.getSampleFrequencies()[3] * epoch))
    ##################################
    f._close()
    loopEpoch = (int)(minN / epoch)
    #print(str(loopEpoch)+' Hr')
    People = [FileName.split('.edf')[0]]
    XML = [f for f in listdir(XMLPath) if f.endswith(".xml") and f == str(FileName.split(".edf")[0])+"-profusion.xml"]
    print(XML)
    fullpath = XMLPath+"\\"+XML[0]
    #print(fullpath)
    DOMTree = xml.dom.minidom.parse(fullpath)
    collection = DOMTree.documentElement
    stagesXML = collection.getElementsByTagName("SleepStage")
    stagesSeparate = []
    stagesALL=[]
    valE = (int)(30/epoch )
    for i in range(len(stagesXML)):
        stagesSeparate.append([(int)(stagesXML[i].childNodes[0].data)]*valE)
    del stagesXML
    for i in range(len(stagesSeparate)):
        for j in range(valE):
            stagesALL.append(stagesSeparate[i][j])
    del stagesSeparate
    for I in range(loopEpoch):

        ALL = [min(SaO2[I]), max(SaO2[I]), np.mean(SaO2[I]), np.std(SaO2[I]), kurtosis(SaO2[I]),
               min(PR[I]), max(PR[I]), np.mean(PR[I]), np.std(PR[I]), kurtosis(PR[I]),
               min(Position[I]), max(Position[I]), np.mean(Position[I]), np.std(Position[I]), kurtosis(Position[I]),
               min(light[I]), max(light[I]), np.mean(light[I]), np.std(light[I]), kurtosis(light[I]),
               min(ox_stat[I]), max(ox_stat[I]), np.mean(ox_stat[I]), np.std(ox_stat[I]), kurtosis(ox_stat[I]),
               min(airflow[I]), max(airflow[I]), np.mean(airflow[I]), np.std(airflow[I]), kurtosis(airflow[I]),
               min(ThorRes[I]), max(ThorRes[I]), np.mean(ThorRes[I]), np.std(ThorRes[I]), kurtosis(ThorRes[I]),
               min(AbdoRes[I]), max(AbdoRes[I]), np.mean(AbdoRes[I]), np.std(AbdoRes[I]), kurtosis(AbdoRes[I]),
               min(EOG_L[I]), max(EOG_L[I]), np.mean(EOG_L[I]), np.std(EOG_L[I]), kurtosis(EOG_L[I]),
               min(EOG_R[I]), max(EOG_R[I]), np.mean(EOG_R[I]), np.std(EOG_R[I]), kurtosis(EOG_R[I]),
               min(EEG_sec[I]), max(EEG_sec[I]), np.mean(EEG_sec[I]), np.std(EEG_sec[I]), kurtosis(EEG_sec[I]),
               min(EEG[I]), max(EEG[I]), np.mean(EEG[I]), np.std(EEG[I]), kurtosis(EEG[I]),
               min(ECG[I]), max(ECG[I]), np.mean(ECG[I]), np.std(ECG[I]), kurtosis(ECG[I]),
               min(EMG[I]), max(EMG[I]), np.mean(EMG[I]), np.std(EMG[I]), kurtosis(EMG[I]),
               stagesALL[I]
               ]
        People.append(ALL)
        del ALL
    #print(len(People))
    string = str(People).replace("'","").replace("|","").replace("[","").replace("]","").split(",")
    del People
    WriteHeaderCSV(string)


############### end section read file EDF ##############


########################  Write CSV File #############################################

def WriteHeaderCSV(listData):
    if os.path.isfile(PathOutputFileName):
        print(" Save to CSV File")
        with open(PathOutputFileName, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(listData)

    else:
        ###Create header og CSV as e1_EEG(sec) mean epoch 1 of signal EEG
        print("Create file")
        s = ['subject']
        Signal = ['SaO2_min', 'SaO2_max', 'SaO2_avg', 'SaO2_SD', 'SaO2_kurtosis',
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
        for n in range((int)(50730 / epoch)):
            for i in Signal:
                s.append('e' + str(n + 1) + '_' + i)

        with open(PathOutputFileName, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(s)
            spamwriter.writerow(listData)
            print("Save to CSV File")



########################################################################################




#########################  Main #######################################################
if __name__ == '__main__':
    ### You must create folder name = "edf" and put file .edf in folder ###
    epoch = (int)(input("Enter epoch do you want (ex. 5,10,15,30): "))
    mypath = input("Enter Directory keep EDF File: ")
    XMLPath = input("Enter path of XML File:")
    FileName = [f for f in listdir(mypath) if f.endswith(".edf")]
    PathOutputFileName = input("Enter path to keep CSV File: ") + "\\"
    OutputFileName = input("Enter output File Name of CSV: ") + ".csv"
    start = datetime.datetime.now()

    PathOutputFileName = PathOutputFileName + OutputFileName
    print(PathOutputFileName)
    countFile = len(FileName)
    i = 1
    for numFile in FileName:
        fullpath = mypath + "\\" + numFile
        print("#" + str(i) + " in " + str(countFile) + "-----------------" + numFile + "-----------------")
        ReadEDFFile(fullpath, numFile)
        i = i + 1
        # WriteHeaderCSV(FileName)
    end = datetime.datetime.now()
    print("start:",start)
    print("end:",end)
    print("total:",end-start)
#########################  End Main  ##################################################

