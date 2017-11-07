import os
import pyedflib
from scipy.stats import kurtosis
import csv
import xml.dom.minidom
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from multiprocessing import Process
np.seterr(divide='ignore', invalid='ignore')
###########  Global Variable ##############
epoch = 30  # Config Epoch as 5s 10s 15s 30s #
PathOutputFileName = ""
normalization = ""
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
###########################################

def chunks(ls, n):
    return np.asarray([np.asarray(ls[i:i + n]) for i in range(0, len(ls), n)])


###################  Section read EDF file ##################
def ReadEDFFile(Path):
    global PathOutputFileName,epoch,normalization
    PathXML = Path.split('\\')[len(Path.split('\\'))-1]
    FileName = Path.split('\\')[len(Path.split('\\'))-1]
    PathXML = Path.replace(PathXML,PathXML.replace('.edf','-profusion.xml'))
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
    print("read XML ",FileName)
    DOMTree = xml.dom.minidom.parse(PathXML)
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
    #print("minECG:",len(minECG),minECG)
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
    del SaO2, PR, Position, light, ox_stat, airflow, ThorRes, AbdoRes
    del EOG_R, EOG_L, EEG_sec, EEG, ECG, EMG
    ############################   Calulate normalization ##########################################
    #print(minECG)
    if normalization=='y' or normalization=='Y':
        minSaO2 =  minmax_scale(minSaO2)
        minPR = minmax_scale(minPR)
        minPosition = minmax_scale(minPosition)
        minLight = minmax_scale(minLight)
        minOX_stat = minmax_scale(minOX_stat)
        minAirflow = minmax_scale(minAirflow)
        minThorRes = minmax_scale(minThorRes)
        minAbdoRes = minmax_scale(minAbdoRes)
        minEOG_L = minmax_scale(minEOG_L)
        minEOG_R = minmax_scale(minEOG_R)
        minEEG_sec = minmax_scale(minEEG_sec)
        minEEG = minmax_scale(minEEG)
        minECG = minmax_scale(minECG)
        minEMG = minmax_scale(minEMG)
        # print("minECG:",len(minECG),minECG)
        maxSaO2 = minmax_scale(maxSaO2)
        maxPR = minmax_scale(maxPR)
        maxPosition = minmax_scale(maxPosition)
        maxLight = minmax_scale(maxLight)
        maxOX_stat = minmax_scale(maxOX_stat)
        maxAirflow = minmax_scale(maxAirflow)
        maxThorRes = minmax_scale(maxThorRes)
        maxAbdoRes = minmax_scale(maxAbdoRes)
        maxEOG_L = minmax_scale(maxEOG_L)
        maxEOG_R = minmax_scale(maxEOG_R)
        maxEEG_sec = minmax_scale(maxEEG_sec)
        maxEEG = minmax_scale(maxEEG)
        maxECG = minmax_scale(maxECG)
        maxEMG = minmax_scale(maxEMG)
        # print(len(maxECG),maxECG)

        avgSaO2 =minmax_scale(avgSaO2)
        avgPR = minmax_scale(avgPR)
        avgPosition = minmax_scale(avgPosition)
        avgLight = minmax_scale(avgLight)
        avgOX_stat = minmax_scale(avgOX_stat)
        avgAirflow = minmax_scale(avgAirflow)
        avgThorRes = minmax_scale(avgThorRes)
        avgAbdoRes = minmax_scale(avgAbdoRes)
        avgEOG_L = minmax_scale(avgEOG_L)
        avgEOG_R = minmax_scale(avgEOG_R)
        avgEEG_sec = minmax_scale(avgEEG_sec)
        avgEEG = minmax_scale(avgEEG)
        avgECG = minmax_scale(avgECG)
        avgEMG = minmax_scale(avgEMG)

        sdSaO2 = minmax_scale(sdSaO2)
        sdPR = minmax_scale(sdPR)
        sdPosition = minmax_scale(sdPosition)
        sdLight = minmax_scale(sdLight)
        sdOX_stat = minmax_scale(sdOX_stat)
        sdAirflow = minmax_scale(sdAirflow)
        sdThorRes = minmax_scale(sdThorRes)
        sdAbdoRes = minmax_scale(sdAbdoRes)
        sdEOG_L = minmax_scale(sdEOG_L)
        sdEOG_R = minmax_scale(sdEOG_R)
        sdEEG_sec = minmax_scale(sdEEG_sec)
        sdEEG = minmax_scale(sdEEG)
        sdECG = minmax_scale(sdECG)
        sdEMG = minmax_scale(sdEMG)

        kurtosisSaO2 = minmax_scale(kurtosisSaO2)
        kurtosisPR = minmax_scale(kurtosisPR)
        kurtosisPosition = minmax_scale(kurtosisPosition)
        kurtosisLight = minmax_scale(kurtosisLight)
        kurtosisOX_stat = minmax_scale(kurtosisOX_stat)
        kurtosisAirflow = minmax_scale(kurtosisAirflow)
        kurtosisThorRes = minmax_scale(kurtosisThorRes)
        kurtosisAbdoRes = minmax_scale(kurtosisAbdoRes)
        kurtosisEOG_L = minmax_scale(kurtosisEOG_L)
        kurtosisEOG_R = minmax_scale(kurtosisEOG_R)
        kurtosisEEG_sec = minmax_scale(kurtosisEEG_sec)
        kurtosisEEG = minmax_scale(kurtosisEEG)
        kurtosisECG = minmax_scale(kurtosisECG)
        kurtosisEMG = minmax_scale(kurtosisEMG)

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


    Frame = pd.DataFrame(People, columns=Signal)
    if os.path.isfile(PathOutputFileName):
        Frame.to_csv(PathOutputFileName, mode='a', index=False, header=False)
    else:
        Frame.to_csv(PathOutputFileName, mode='w', index=False)

############### end section read file EDF ##############

#########################  Main #######################################################
def Start():
    global PathOutputFileName,normalization,epoch
    print("#### This program is preprocessing data ######")
    print("You must to keep EDF file and XML file in same Directory!!!!")
    ### You must create folder name = "edf" and put file .edf in folder ###
    start = datetime.datetime.now()
    epoch = (int)(input("Enter epoch do you want (ex. 5,10,15,30): "))
    mypath = input("Enter Directory keep EDF and XML File: ")

    FileName = [f for f in os.listdir(mypath) if f.endswith(".edf")]
    PathOutputFileName = input("Enter path to keep CSV File: ") + "\\"
    OutputFileName=''
    normalization = input("You need to normalization (Y/N):")
    if normalization=='Y' or normalization=='y':
        OutputFileName = 'SeparateEachEpochPerRow_Signal_'+str(epoch)+'s_Normalization.csv'
    else:
        OutputFileName = 'SeparateEachEpochPerRow_Signal_'+str(epoch)+'s_Unnormalization.csv'

    PathOutputFileName = PathOutputFileName + OutputFileName
    
    countFile = len(FileName)

    for i,File in enumerate(FileName):
        fullpath = mypath + "\\" + File
        print("#" + str(i+1) + " in " + str(countFile) + "-----------------" + File + "-----------------")
        ReadEDFFile(fullpath)



    end = datetime.datetime.now()
    print("start:",start)
    print("end:",end)
    print("total:",end-start)
    print(PathOutputFileName)
#########################  End Main  ##################################################
if __name__ == '__main__':
    Start()
