import os
import pyedflib
from scipy.stats import kurtosis
import csv
import xml.dom.minidom
import datetime
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
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
    People = []
    XML = [f for f in os.listdir(XMLPath) if f.endswith(".xml") and f == str(FileName.split(".edf")[0])+"-profusion.xml"]
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
    ######################   min max avf std kurtosis stroe in variable  ##############################

    minSaO2=[min(f) for f in SaO2]
    minPR=[min(f) for f in PR]
    minPosition =[min(f) for f in Position]
    minLight =[min(f) for f in light]
    minOX_stat =[min(f) for f in ox_stat]
    minAirflow  =[min(f) for f in airflow]
    minThorRes =[min(f) for f in ThorRes]
    minAbdoRes =[min(f) for f in AbdoRes]
    minEOG_L=[min(f) for f in EOG_L]
    minEOG_R=[min(f) for f in EOG_R]
    minEEG_sec=[min(f) for f in EEG_sec]
    minEEG=[min(f) for f in EEG]
    minECG=[min(f) for f in ECG]
    minEMG=[min(f) for f in EMG]

    maxSaO2=[max(f) for f in SaO2]
    maxPR=[max(f) for f in PR]
    maxPosition=[max(f) for f in Position]
    maxLight=[max(f) for f in light]
    maxOX_stat=[max(f) for f in ox_stat]
    maxAirflow =[max(f) for f in airflow]
    maxThorRes=[max(f) for f in ThorRes]
    maxAbdoRes = [max(f) for f in AbdoRes]
    maxEOG_L=[max(f) for f in EOG_L]
    maxEOG_R=[max(f) for f in EOG_R]
    maxEEG_sec=[max(f) for f in EEG_sec]
    maxEEG=[max(f) for f in EEG]
    maxECG=[max(f) for f in ECG]
    maxEMG = [max(f) for f in EMG]
    #print(len(maxECG),maxECG)

    avgSaO2=[np.mean(f) for f in SaO2]
    avgPR=[np.mean(f) for f in PR]
    avgPosition=[np.mean(f) for f in Position]
    avgLight=[np.mean(f) for f in light]
    avgOX_stat=[np.mean(f) for f in ox_stat]
    avgAirflow=[np.mean(f) for f in airflow]
    avgThorRes=[np.mean(f) for f in ThorRes]
    avgAbdoRes = [np.mean(f) for f in AbdoRes]
    avgEOG_L=[np.mean(f) for f in EOG_L]
    avgEOG_R=[np.mean(f) for f in EOG_R]
    avgEEG_sec=[np.mean(f) for f in EEG_sec]
    avgEEG=[np.mean(f) for f in EEG]
    avgECG=[np.mean(f) for f in ECG]
    avgEMG = [np.mean(f) for f in EMG]

    sdSaO2=[np.std(f) for f in SaO2]
    sdPR=[np.std(f) for f in PR]
    sdPosition=[np.std(f) for f in Position]
    sdLight=[np.std(f) for f in light]
    sdOX_stat=[np.std(f) for f in ox_stat]
    sdAirflow=[np.std(f) for f in airflow]
    sdThorRes=[np.std(f) for f in ThorRes]
    sdAbdoRes = [np.std(f) for f in AbdoRes]
    sdEOG_L=[np.std(f) for f in EOG_L]
    sdEOG_R=[np.std(f) for f in EOG_R]
    sdEEG_sec=[np.std(f) for f in EEG_sec]
    sdEEG=[np.std(f) for f in EEG]
    sdECG=[np.std(f) for f in ECG]
    sdEMG = [np.std(f) for f in EMG]

    kurtosisSaO2=[kurtosis(f) for f in SaO2]
    kurtosisPR=[kurtosis(f) for f in PR]
    kurtosisPosition=[kurtosis(f) for f in Position]
    kurtosisLight=[kurtosis(f) for f in light]
    kurtosisOX_stat=[kurtosis(f) for f in ox_stat]
    kurtosisAirflow=[kurtosis(f) for f in airflow]
    kurtosisThorRes=[kurtosis(f) for f in ThorRes]
    kurtosisAbdoRes = [kurtosis(f) for f in AbdoRes]
    kurtosisEOG_L=[kurtosis(f) for f in EOG_L]
    kurtosisEOG_R=[kurtosis(f) for f in EOG_R]
    kurtosisEEG_sec=[kurtosis(f) for f in EEG_sec]
    kurtosisEEG=[kurtosis(f) for f in EEG]
    kurtosisECG=[kurtosis(f) for f in ECG]
    kurtosisEMG = [kurtosis(f) for f in EMG]

    ############################   Calulate normalization ##########################################
    minSaO2 = (minSaO2 - min(minSaO2)) / (max(minSaO2) - min(minSaO2))
    maxSaO2 = (maxSaO2 - min(maxSaO2)) / (max(maxSaO2) - min(maxSaO2))
    avgSaO2 = (avgSaO2 - min(avgSaO2)) / (max(avgSaO2) - min(avgSaO2))
    sdSaO2 = (sdSaO2 - min(sdSaO2)) / (max(sdSaO2) - min(sdSaO2))
    kurtosisSaO2 = np.round((np.asarray(kurtosisSaO2) - min(kurtosisSaO2)) / (max(kurtosisSaO2) - min(kurtosisSaO2)),10)

    minPR = (minPR - min(minPR)) / (max(minPR) - min(minPR))
    maxPR = (maxPR - min(maxPR)) / (max(maxPR) - min(maxPR))
    avgPR = (avgPR - min(avgPR)) / (max(avgPR) - min(avgPR))
    sdPR = (sdPR - min(sdPR)) / (max(sdPR) - min(sdPR))
    kurtosisPR = np.round((np.asarray(kurtosisPR) - min(kurtosisPR)) / (max(kurtosisPR) - min(kurtosisPR)),10)

    minPosition = (minPosition - min(minPosition)) / (max(minPosition) - min(minPosition))
    maxPosition = (maxPosition - min(maxPosition)) / (max(maxPosition) - min(maxPosition))
    avgPosition = (avgPosition - min(avgPosition)) / (max(avgPosition) - min(avgPosition))
    sdPosition = (sdPosition - min(sdPosition)) / (max(sdPosition) - min(sdPosition))
    kurtosisPosition = np.round((np.asarray(kurtosisPosition) - min(kurtosisPosition)) / (max(kurtosisPosition) - min(kurtosisPosition)),10)

    minLight = (minLight - min(minLight)) / (max(minLight) - min(minLight))
    maxLight = (maxLight - min(maxLight)) / (max(maxLight) - min(maxLight))
    avgLight = (avgLight - min(avgLight)) / (max(avgLight) - min(avgLight))
    sdLight = (sdLight - min(sdLight)) / (max(sdLight) - min(sdLight))
    kurtosisLight = np.round((np.asarray(kurtosisLight) - min(kurtosisLight)) / (max(kurtosisLight) - min(kurtosisLight)),10)

    minOX_stat = (minOX_stat - min(minOX_stat)) / (max(minOX_stat) - min(minOX_stat))
    maxOX_stat = (maxOX_stat - min(maxOX_stat)) / (max(maxOX_stat) - min(maxOX_stat))
    avgOX_stat = (avgOX_stat - min(avgOX_stat)) / (max(avgOX_stat) - min(avgOX_stat))
    sdOX_stat = (sdOX_stat - min(sdOX_stat)) / (max(sdOX_stat) - min(sdOX_stat))
    kurtosisOX_stat = np.round((np.asarray(kurtosisOX_stat) - min(kurtosisOX_stat)) / (max(kurtosisOX_stat) - min(kurtosisOX_stat)),10)

    minAirflow = (minAirflow - min(minAirflow)) / (max(minAirflow) - min(minAirflow))
    maxAirflow = (maxAirflow - min(maxAirflow)) / (max(maxAirflow) - min(maxAirflow))
    avgAirflow = (avgAirflow - min(avgAirflow)) / (max(avgAirflow) - min(avgAirflow))
    sdAirflow = (sdAirflow - min(sdAirflow)) / (max(sdAirflow) - min(sdAirflow))
    kurtosisAirflow = np.round((np.asarray(kurtosisAirflow) - min(kurtosisAirflow)) / (max(kurtosisAirflow) - min(kurtosisAirflow)),10)

    minThorRes = (minThorRes - min(minThorRes)) / (max(minThorRes) - min(minThorRes))
    maxThorRes = (maxThorRes - min(maxThorRes)) / (max(maxThorRes) - min(maxThorRes))
    avgThorRes = (avgThorRes - min(avgThorRes)) / (max(avgThorRes) - min(avgThorRes))
    sdThorRes = (sdThorRes - min(sdThorRes)) / (max(sdThorRes) - min(sdThorRes))
    kurtosisThorRes = np.round((np.asarray(kurtosisThorRes) - min(kurtosisThorRes)) / (max(kurtosisThorRes) - min(kurtosisThorRes)),10)

    minAbdoRes = (minAbdoRes - min(minAbdoRes)) / (max(minAbdoRes) - min(minAbdoRes))
    maxAbdoRes = (maxAbdoRes - min(maxAbdoRes)) / (max(maxAbdoRes) - min(maxAbdoRes))
    avgAbdoRes = (avgAbdoRes - min(avgAbdoRes)) / (max(avgAbdoRes) - min(avgAbdoRes))
    sdAbdoRes = (sdAbdoRes - min(sdAbdoRes)) / (max(sdAbdoRes) - min(sdAbdoRes))
    kurtosisAbdoRes = np.round((np.asarray(kurtosisAbdoRes) - min(kurtosisAbdoRes)) / (max(kurtosisAbdoRes) - min(kurtosisAbdoRes)),10)

    minEOG_L = (minEOG_L - min(minEOG_L)) / (max(minEOG_L) - min(minEOG_L))
    maxEOG_L = (maxEOG_L - min(maxEOG_L)) / (max(maxEOG_L) - min(maxEOG_L))
    avgEOG_L = (avgEOG_L - min(avgEOG_L)) / (max(avgEOG_L) - min(avgEOG_L))
    sdEOG_L = (sdEOG_L - min(sdEOG_L)) / (max(sdEOG_L) - min(sdEOG_L))
    kurtosisEOG_L = np.round((np.asarray(kurtosisEOG_L) - min(kurtosisEOG_L)) / (max(kurtosisEOG_L) - min(kurtosisEOG_L)),10)

    minEOG_R = (minEOG_R - min(minEOG_R)) / (max(minEOG_R) - min(minEOG_R))
    maxEOG_R = (maxEOG_R - min(maxEOG_R)) / (max(maxEOG_R) - min(maxEOG_R))
    avgEOG_R = (avgEOG_R - min(avgEOG_R)) / (max(avgEOG_R) - min(avgEOG_R))
    sdEOG_R = (sdEOG_R - min(sdEOG_R)) / (max(sdEOG_R) - min(sdEOG_R))
    kurtosisEOG_R = np.round((np.asarray(kurtosisEOG_R) - min(kurtosisEOG_R)) / (max(kurtosisEOG_R) - min(kurtosisEOG_R)),10)

    minEEG_sec = (minEEG_sec - min(minEEG_sec)) / (max(minEEG_sec) - min(minEEG_sec))
    maxEEG_sec = (maxEEG_sec - min(maxEEG_sec)) / (max(maxEEG_sec) - min(maxEEG_sec))
    avgEEG_sec = (avgEEG_sec - min(avgEEG_sec)) / (max(avgEEG_sec) - min(avgEEG_sec))
    sdEEG_sec = (sdEEG_sec - min(sdEEG_sec)) / (max(sdEEG_sec) - min(sdEEG_sec))
    kurtosisEEG_sec = np.round((np.asarray(kurtosisEEG_sec) - min(kurtosisEEG_sec)) / (max(kurtosisEEG_sec) - min(kurtosisEEG_sec)),10)

    minEEG = (minEEG - min(minEEG)) / (max(minEEG) - min(minEEG))
    maxEEG = (maxEEG - min(maxEEG)) / (max(maxEEG) - min(maxEEG))
    avgEEG = (avgEEG - min(avgEEG)) / (max(avgEEG) - min(avgEEG))
    sdEEG = (sdEEG - min(sdEEG)) / (max(sdEEG) - min(sdEEG))
    kurtosisEEG = np.round((np.asarray(kurtosisEEG) - min(kurtosisEEG)) / (max(kurtosisEEG) - min(kurtosisEEG)),10)

    minECG = (minECG - min(minECG)) / (max(minECG) - min(minECG))
    if max(maxECG)-min(maxECG) ==0 :
        maxECG = maxECG/max(maxECG)
    else:
        maxECG = (maxECG - min(maxECG)) / (max(maxECG) - min(maxECG))

    avgECG = (avgECG - min(avgECG)) / (max(avgECG) - min(avgECG))
    sdECG = (sdECG - min(sdECG)) / (max(sdECG) - min(sdECG))
    kurtosisECG = np.round((np.asarray(kurtosisECG) - min(kurtosisECG)) / (max(kurtosisECG) - min(kurtosisECG)),10)

    minEMG = (minEMG - min(minEMG)) / (max(minEMG) - min(minEMG))
    maxEMG = (maxEMG - min(maxEMG)) / (max(maxEMG) - min(maxEMG))
    avgEMG = (avgEMG - min(avgEMG)) / (max(avgEMG) - min(avgEMG))
    sdEMG = (sdEMG - min(sdEMG)) / (max(sdEMG) - min(sdEMG))
    kurtosisECG = np.round((np.asarray(kurtosisEMG) - min(kurtosisEMG)) / (max(kurtosisEMG) - min(kurtosisEMG)),10)


    ######################################################################
    for I in range(loopEpoch):

        ALL = [FileName.split('.edf')[0],I+1,
               minSaO2[I], maxSaO2[I], avgSaO2[I], sdSaO2[I], kurtosisSaO2[I],
               minPR[I], maxPR[I], avgPR[I], sdPR[I], kurtosisPR[I],
               minPosition[I], maxPosition[I], avgPosition[I], sdPosition[I], kurtosisPosition[I],
               minLight[I], maxLight[I], avgLight[I], sdLight[I], kurtosisLight[I],
               minOX_stat[I], maxOX_stat[I], avgOX_stat[I], sdOX_stat[I], kurtosisOX_stat[I],
               minAirflow[I], maxAirflow[I], avgAirflow[I], sdAirflow[I], kurtosisAirflow[I],
               minThorRes[I], maxThorRes[I], avgThorRes[I], sdThorRes[I], kurtosisThorRes[I],
               minAbdoRes[I], maxAbdoRes[I], avgAbdoRes[I], sdAbdoRes[I], kurtosisAbdoRes[I],
               minEOG_L[I], maxEOG_L[I], avgEOG_L[I], sdEOG_L[I], kurtosisEOG_L[I],
               minEOG_R[I], maxEOG_R[I], avgEOG_R[I], sdEOG_R[I], kurtosisEOG_R[I],
               minEEG_sec[I], maxEEG_sec[I], avgEEG_sec[I], sdEEG_sec[I], kurtosisEEG_sec[I],
               minEEG[I], maxEEG[I], avgEEG[I], sdEEG[I], kurtosisEEG[I],
               minECG[I], maxECG[I], avgECG[I], sdECG[I], kurtosisECG[I],
               minEMG[I], maxEMG[I], avgEMG[I], sdEMG[I], kurtosisEMG[I],
               stagesALL[I]
               ]
        People.append(ALL)
        ALL=[]
    del SaO2,PR,Position,light,ox_stat,airflow,ThorRes,AbdoRes
    del EOG_R,EOG_L,EEG_sec,EEG,ECG,EMG

    for i in People:
        WriteHeaderCSV(i)

    del People

############### end section read file EDF ##############


########################  Write CSV File #############################################

def WriteHeaderCSV(listData):
    if os.path.isfile(PathOutputFileName):
        #print(" Save to CSV File")
        with open(PathOutputFileName, 'a' , newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(listData)

    else:
        ###Create header og CSV as e1_EEG(sec) mean epoch 1 of signal EEG
        print("Create file")

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


        with open(PathOutputFileName, 'w',newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(Signal)
            spamwriter.writerow(listData)
            print("Save to CSV File")



########################################################################################




#########################  Main #######################################################
if __name__ == '__main__':
    ### You must create folder name = "edf" and put file .edf in folder ###
	#print("This program is separate each epoch to each row and ####  Normalization Data #####")
    epoch = (int)(input("Enter epoch do you want (ex. 5,10,15,30): "))
    mypath = input("Enter Directory keep EDF File: ")
    XMLPath = input("Enter path of XML File:")
    FileName = [f for f in os.listdir(mypath) if f.endswith(".edf")]
    PathOutputFileName = input("Enter path to keep CSV File: ") + "\\"
    OutputFileName = "SeparateEachEpochPerRow_Signal_"+str(epoch)+"s_normalization.csv"
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

