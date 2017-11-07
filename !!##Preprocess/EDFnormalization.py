from os import listdir
from os.path import isfile, join
import os.path
import pyedflib
import numpy as np
from scipy.stats import kurtosis
import csv
import xml.dom.minidom
import datetime
import numpy as np
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
    ######################   normalization  ##############################

    minSaO2=[]
    minPR=[]
    minPosition =[]
    minLight =[]
    minOX_stat =[]
    minAirflow  =[]
    minThorRes =[]
    minAbdoRes =[]
    minEOG_L=[]
    minEOG_R=[]
    minEEG_sec=[]
    minEEG=[]
    minECG=[]
    minEMG=[]

    maxSaO2=[]
    maxPR=[]
    maxPosition=[]
    maxLight=[]
    maxOX_stat=[]
    maxAirflow =[]
    maxThorRes=[]
    maxAbdoRes = []
    maxEOG_L=[]
    maxEOG_R=[]
    maxEEG_sec=[]
    maxEEG=[]
    maxECG=[]
    maxEMG = []


    avgSaO2=[]
    avgPR=[]
    avgPosition=[]
    avgLight=[]
    avgOX_stat=[]
    avgAirflow=[]
    avgThorRes=[]
    avgAbdoRes = []
    avgEOG_L=[]
    avgEOG_R=[]
    avgEEG_sec=[]
    avgEEG=[]
    avgECG=[]
    avgEMG = []

    sdSaO2=[]
    sdPR=[]
    sdPosition=[]
    sdLight=[]
    sdOX_stat=[]
    sdAirflow=[]
    sdThorRes=[]
    sdAbdoRes = []
    sdEOG_L=[]
    sdEOG_R=[]
    sdEEG_sec=[]
    sdEEG=[]
    sdECG=[]
    sdEMG = []

    kurtosisSaO2=[]
    kurtosisPR=[]
    kurtosisPosition=[]
    kurtosisLight=[]
    kurtosisOX_stat=[]
    kurtosisAirflow=[]
    kurtosisThorRes=[]
    kurtosisAbdoRes = []
    kurtosisEOG_L=[]
    kurtosisEOG_R=[]
    kurtosisEEG_sec=[]
    kurtosisEEG=[]
    kurtosisECG=[]
    kurtosisEMG = []
    '''
    for i in range(loopEpoch):
        minSaO2.append(min(SaO2[i]))
        minPR.append(min(PR[i]))
        minPosition.append(min(Position[i]))
        minLight.append(min(light[i]))
        minOX_stat.append(min(ox_stat[i]))
        minAirflow.append(min(airflow[i]))
        minThorRes.append(min(ThorRes[i]))
        minAbdoRes.append(min(AbdoRes[i]))
        minEOG_L.append(min(EOG_L[i]))
        minEOG_R.append(min(EOG_R[i]))
        minEEG_sec.append(min(EEG_sec[i]))
        minEEG.append(min(EEG[i]))
        minECG.append(min(ECG[i]))
        minEMG.append(min(EMG[i]))
        maxEEG_sec.append(max(EEG_sec[i]))
        maxEEG.append(max(EEG[i]))
        maxECG.append(max(ECG[i]))
        maxEMG.append(max(EMG[i]))

        avgSaO2.append(np.mean(SaO2[i]))
        avgPR.append(np.mean(PR[i]))
        avgPosition.append(np.mean(Position[i]))

        maxSaO2.append(max(SaO2[i]))
        maxPR.append(max(PR[i]))
        maxPosition.append(max(Position[i]))
        maxLight.append(max(light[i]))
        maxOX_stat.append(max(ox_stat[i]))
        maxAirflow.append(max(airflow[i]))
        maxThorRes.append(max(ThorRes[i]))
        maxAbdoRes.append(max(AbdoRes[i]))
        maxEOG_L.append(max(EOG_L[i]))
        maxEOG_R.append(max(EOG_R[i]))
        avgLight.append(np.mean(light[i]))
        avgOX_stat.append(np.mean(ox_stat[i]))
        avgAirflow.append(np.mean(airflow[i]))
        avgThorRes.append(np.mean(ThorRes[i]))
        avgAbdoRes.append(np.mean(AbdoRes[i]))
        avgEOG_L.append(np.mean(EOG_L[i]))
        avgEOG_R.append(np.mean(EOG_R[i]))
        avgEEG_sec.append(np.mean(EEG_sec[i]))
        avgEEG.append(np.mean(EEG[i]))
        avgECG.append(np.mean(ECG[i]))
        avgEMG.append(np.mean(EMG[i]))

        sdSaO2.append(np.std(SaO2[i]))
        sdPR.append(np.std(PR[i]))
        sdPosition.append(np.std(Position[i]))
        sdLight.append(np.std(light[i]))
        sdOX_stat.append(np.std(ox_stat[i]))
        sdAirflow.append(np.std(airflow[i]))
        sdThorRes.append(np.std(ThorRes[i]))
        sdAbdoRes.append(np.std(AbdoRes[i]))
        sdEOG_L.append(np.std(EOG_L[i]))
        sdEOG_R.append(np.std(EOG_R[i]))
        sdEEG_sec.append(np.std(EEG_sec[i]))
        sdEEG.append(np.std(EEG[i]))
        sdECG.append(np.std(ECG[i]))
        sdEMG.append(np.std(EMG[i]))

        kurtosisSaO2.append(kurtosis(SaO2[i]))
        kurtosisPR.append(kurtosis(PR[i]))
        kurtosisPosition.append(kurtosis(Position[i]))
        kurtosisLight.append(kurtosis(light[i]))
        kurtosisOX_stat.append(kurtosis(ox_stat[i]))
        kurtosisAirflow.append(kurtosis(airflow[i]))
        kurtosisThorRes.append(kurtosis(ThorRes[i]))
        kurtosisAbdoRes.append(kurtosis(AbdoRes[i]))
        kurtosisEOG_L.append(kurtosis(EOG_L[i]))
        kurtosisEOG_R.append(kurtosis(EOG_R[i]))
        kurtosisEEG_sec.append(kurtosis(EEG_sec[i]))
        kurtosisEEG.append(kurtosis(EEG[i]))
        kurtosisECG.append(kurtosis(ECG[i]))
        kurtosisEMG.append(kurtosis(EMG[i]))
    max(minSaO2)
    '''
    ######################################################################
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
    del SaO2,PR,Position,light,ox_stat,airflow,ThorRes,AbdoRes
    del EOG_R,EOG_L,EEG_sec,EEG,ECG,EMG

    for I in range(1,len(People)):
        for J in range(70):
            if J==0 :  minSaO2.append(People[I][J])
            if J==1 :  maxSaO2.append(People[I][J])
            if J==2 :  avgSaO2.append(People[I][J])
            if J==3 :  sdSaO2.append(People[I][J])
            if J == 4: kurtosisSaO2.append(round(People[I][J],10))

            if J == 5: minPR.append(People[I][J])
            if J == 6: maxPR.append(People[I][J])
            if J == 7: avgPR.append(People[I][J])
            if J == 8: sdPR.append(People[I][J])
            if J == 9: kurtosisPR.append(round(People[I][J],10))

            if J==10 :  minPosition.append(People[I][J])
            if J==11 :  maxPosition.append(People[I][J])
            if J==12 :  avgPosition.append(People[I][J])
            if J==13 :  sdPosition.append(People[I][J])
            if J == 14: kurtosisPosition.append(round(People[I][J],10))

            if J==15 :  minLight.append(People[I][J])
            if J==16 :  maxLight.append(People[I][J])
            if J==17 :  avgLight.append(People[I][J])
            if J==18 :  sdLight.append(People[I][J])
            if J == 19: kurtosisLight.append(round(People[I][J],10))

            if J==20 :  minOX_stat.append(People[I][J])
            if J==21 :  maxOX_stat.append(People[I][J])
            if J==22 :  avgOX_stat.append(People[I][J])
            if J==23 :  sdOX_stat.append(People[I][J])
            if J ==24: kurtosisOX_stat.append(round(People[I][J],10))

            if J==25 :  minAirflow.append(People[I][J])
            if J==26 :  maxAirflow.append(People[I][J])
            if J==27 :  avgAirflow.append(People[I][J])
            if J==28 :  sdAirflow.append(People[I][J])
            if J == 29: kurtosisAirflow.append(round(People[I][J],10))

            if J==30 :  minThorRes.append(People[I][J])
            if J==31 :  maxThorRes.append(People[I][J])
            if J==32 :  avgThorRes.append(People[I][J])
            if J==33 :  sdThorRes.append(People[I][J])
            if J ==34:  kurtosisThorRes.append(round(People[I][J],10))

            if J==35 :  minAbdoRes.append(People[I][J])
            if J==36 :  maxAbdoRes.append(People[I][J])
            if J==37 :  avgAbdoRes.append(People[I][J])
            if J==38:   sdAbdoRes.append(People[I][J])
            if J ==39 : kurtosisAbdoRes.append(round(People[I][J],10))

            if J==40 :  minEOG_L.append(People[I][J])
            if J==41 :  maxEOG_L.append(People[I][J])
            if J==42 :  avgEOG_L.append(People[I][J])
            if J==43 :  sdEOG_L.append(People[I][J])
            if J ==44: kurtosisEOG_L.append(round(People[I][J],10))

            if J==45 :  minEOG_R.append(People[I][J])
            if J==46:   maxEOG_R.append(People[I][J])
            if J==47 :  avgEOG_R.append(People[I][J])
            if J==48 :  sdEOG_R.append(People[I][J])
            if J == 49: kurtosisEOG_R.append(round(People[I][J],10))

            if J==50 :  minEEG_sec.append(People[I][J])
            if J==51 :  maxEEG_sec.append(People[I][J])
            if J==52 :  avgEEG_sec.append(People[I][J])
            if J==53 :  sdEEG_sec.append(People[I][J])
            if J ==54:  kurtosisEEG_sec.append(round(People[I][J],10))

            if J==55 :  minEEG.append(People[I][J])
            if J==56:   maxEEG.append(People[I][J])
            if J==57:   avgEEG.append(People[I][J])
            if J==58 :  sdEEG.append(People[I][J])
            if J ==59:  kurtosisEEG.append(round(People[I][J],10))

            if J==60 :  minECG.append(People[I][J])
            if J==61 :  maxECG.append(People[I][J])
            if J==62 :  avgECG.append(People[I][J])
            if J==63 :  sdECG.append(People[I][J])
            if J ==64:  kurtosisECG.append(round(People[I][J],10))

            if J==65 :  minEMG.append(People[I][J])
            if J==66 :  maxEMG.append(People[I][J])
            if J==67 :  avgEMG.append(People[I][J])
            if J==68 :  sdEMG.append(People[I][J])
            if J ==69:  kurtosisEMG.append(round(People[I][J],10))


    ############################   Calulate normalization ##########################################
    minSaO2 = (minSaO2-min(minSaO2))/(max(minSaO2)-min(minSaO2))
    maxSaO2 = (maxSaO2 - min(maxSaO2)) / (max(maxSaO2) - min(maxSaO2))
    avgSaO2 = (avgSaO2 - min(avgSaO2)) / (max(avgSaO2) - min(avgSaO2))
    sdSaO2 = (sdSaO2 - min(sdSaO2)) / (max(sdSaO2) - min(sdSaO2))
    kurtosisSaO2 = (np.asarray(kurtosisSaO2) - min(kurtosisSaO2)) / (max(kurtosisSaO2) - min(kurtosisSaO2))


    minPR = (minPR - min(minPR)) / (max(minPR) - min(minPR))
    maxPR = (maxPR - min(maxPR)) / (max(maxPR) - min(maxPR))
    avgPR = (avgPR - min(avgPR)) / (max(avgPR) - min(avgPR))
    sdPR = (sdPR - min(sdPR)) / (max(sdPR) - min(sdPR))
    kurtosisPR = (np.asarray(kurtosisPR) - min(kurtosisPR)) / (max(kurtosisPR) - min(kurtosisPR))


    minPosition = (minPosition - min(minPosition)) / (max(minPosition) - min(minPosition))
    maxPosition = (maxPosition - min(maxPosition)) / (max(maxPosition) - min(maxPosition))
    avgPosition = (avgPosition - min(avgPosition)) / (max(avgPosition) - min(avgPosition))
    sdPosition = (sdPosition - min(sdPosition)) / (max(sdPosition) - min(sdPosition))
    kurtosisPosition = (np.asarray(kurtosisPosition) - min(kurtosisPosition)) / (max(kurtosisPosition) - min(kurtosisPosition))

    minLight = (minLight - min(minLight)) / (max(minLight) - min(minLight))
    maxLight = (maxLight - min(maxLight)) / (max(maxLight) - min(maxLight))
    avgLight = (avgLight - min(avgLight)) / (max(avgLight) - min(avgLight))
    sdLight = (sdLight - min(sdLight)) / (max(sdLight) - min(sdLight))
    kurtosisLight = (np.asarray(kurtosisLight) - min(kurtosisLight)) / (max(kurtosisLight) - min(kurtosisLight))

    minOX_stat = (minOX_stat - min(minOX_stat)) / (max(minOX_stat) - min(minOX_stat))
    maxOX_stat = (maxOX_stat - min(maxOX_stat)) / (max(maxOX_stat) - min(maxOX_stat))
    avgOX_stat = (avgOX_stat - min(avgOX_stat)) / (max(avgOX_stat) - min(avgOX_stat))
    sdOX_stat = (sdOX_stat - min(sdOX_stat)) / (max(sdOX_stat) - min(sdOX_stat))
    kurtosisOX_stat = (np.asarray(kurtosisOX_stat) - min(kurtosisOX_stat)) / (max(kurtosisOX_stat) - min(kurtosisOX_stat))

    minAirflow = (minAirflow - min(minAirflow)) / (max(minAirflow) - min(minAirflow))
    maxAirflow = (maxAirflow - min(maxAirflow)) / (max(maxAirflow) - min(maxAirflow))
    avgAirflow = (avgAirflow - min(avgAirflow)) / (max(avgAirflow) - min(avgAirflow))
    sdAirflow = (sdAirflow - min(sdAirflow)) / (max(sdAirflow) - min(sdAirflow))
    kurtosisAirflow = (np.asarray(kurtosisAirflow) - min(kurtosisAirflow)) / (max(kurtosisAirflow) - min(kurtosisAirflow))

    minThorRes = (minThorRes - min(minThorRes)) / (max(minThorRes) - min(minThorRes))
    maxThorRes = (maxThorRes - min(maxThorRes)) / (max(maxThorRes) - min(maxThorRes))
    avgThorRes = (avgThorRes - min(avgThorRes)) / (max(avgThorRes) - min(avgThorRes))
    sdThorRes = (sdThorRes - min(sdThorRes)) / (max(sdThorRes) - min(sdThorRes))
    kurtosisThorRes = (np.asarray(kurtosisThorRes) - min(kurtosisThorRes)) / (max(kurtosisThorRes) - min(kurtosisThorRes))

    minAbdoRes = (minAbdoRes - min(minAbdoRes)) / (max(minAbdoRes) - min(minAbdoRes))
    maxAbdoRes = (maxAbdoRes - min(maxAbdoRes)) / (max(maxAbdoRes) - min(maxAbdoRes))
    avgAbdoRes = (avgAbdoRes - min(avgAbdoRes)) / (max(avgAbdoRes) - min(avgAbdoRes))
    sdAbdoRes = (sdAbdoRes - min(sdAbdoRes)) / (max(sdAbdoRes) - min(sdAbdoRes))
    kurtosisAbdoRes = (np.asarray(kurtosisAbdoRes) - min(kurtosisAbdoRes)) / (max(kurtosisAbdoRes) - min(kurtosisAbdoRes))

    minEOG_L = (minEOG_L - min(minEOG_L)) / (max(minEOG_L) - min(minEOG_L))
    maxEOG_L = (maxEOG_L - min(maxEOG_L)) / (max(maxEOG_L) - min(maxEOG_L))
    avgEOG_L = (avgEOG_L - min(avgEOG_L)) / (max(avgEOG_L) - min(avgEOG_L))
    sdEOG_L = (sdEOG_L - min(sdEOG_L)) / (max(sdEOG_L) - min(sdEOG_L))
    kurtosisEOG_L = (np.asarray(kurtosisEOG_L) - min(kurtosisEOG_L)) / (max(kurtosisEOG_L) - min(kurtosisEOG_L))

    minEOG_R = (minEOG_R - min(minEOG_R)) / (max(minEOG_R) - min(minEOG_R))
    maxEOG_R = (maxEOG_R - min(maxEOG_R)) / (max(maxEOG_R) - min(maxEOG_R))
    avgEOG_R = (avgEOG_R - min(avgEOG_R)) / (max(avgEOG_R) - min(avgEOG_R))
    sdEOG_R = (sdEOG_R - min(sdEOG_R)) / (max(sdEOG_R) - min(sdEOG_R))
    kurtosisEOG_R = (np.asarray(kurtosisEOG_R) - min(kurtosisEOG_R)) / (max(kurtosisEOG_R) - min(kurtosisEOG_R))

    minEEG_sec = (minEEG_sec - min(minEEG_sec)) / (max(minEEG_sec) - min(minEEG_sec))
    maxEEG_sec = (maxEEG_sec - min(maxEEG_sec)) / (max(maxEEG_sec) - min(maxEEG_sec))
    avgEEG_sec = (avgEEG_sec - min(avgEEG_sec)) / (max(avgEEG_sec) - min(avgEEG_sec))
    sdEEG_sec = (sdEEG_sec - min(sdEEG_sec)) / (max(sdEEG_sec) - min(sdEEG_sec))
    kurtosisEEG_sec = (np.asarray(kurtosisEEG_sec) - min(kurtosisEEG_sec)) / (max(kurtosisEEG_sec) - min(kurtosisEEG_sec))

    minEEG = (minEEG - min(minEEG)) / (max(minEEG) - min(minEEG))
    maxEEG = (maxEEG - min(maxEEG)) / (max(maxEEG) - min(maxEEG))
    avgEEG = (avgEEG- min(avgEEG)) / (max(avgEEG) - min(avgEEG))
    sdEEG = (sdEEG - min(sdEEG)) / (max(sdEEG) - min(sdEEG))
    kurtosisEEG = (np.asarray(kurtosisEEG) - min(kurtosisEEG)) / (max(kurtosisEEG) - min(kurtosisEEG))

    minECG = (minECG - min(minECG)) / (max(minECG) - min(minECG))
    maxECG = (maxECG - min(maxECG)) / (max(maxECG) - min(maxECG))
    avgECG = (avgECG - min(avgECG)) / (max(avgECG) - min(avgECG))
    sdECG = (sdECG - min(sdECG)) / (max(sdECG) - min(sdECG))
    kurtosisECG = (np.asarray(kurtosisECG) - min(kurtosisECG)) / (max(kurtosisECG) - min(kurtosisECG))

    minEMG = (minEMG - min(minEMG)) / (max(minEMG) - min(minEMG))
    maxEMG = (maxEMG - min(maxEMG)) / (max(maxEMG) - min(maxEMG))
    avgEMG = (avgEMG - min(avgEMG)) / (max(avgEMG) - min(avgEMG))
    sdEMG = (sdEMG - min(sdEMG)) / (max(sdEMG) - min(sdEMG))
    kurtosisECG = (np.asarray(kurtosisEMG) - min(kurtosisEMG)) / (max(kurtosisEMG) - min(kurtosisEMG))

    for I in range(1, len(People)):
        for J in range(70):
            if J == 0:  People[I][J] = minSaO2[I-1]
            if J == 1:  People[I][J] =maxSaO2[I-1]
            if J == 2:  People[I][J]=avgSaO2[I-1]
            if J == 3:  People[I][J]=sdSaO2[I-1]
            if J == 4:  People[I][J]=kurtosisSaO2[I-1]

            if J == 5: People[I][J] = minPR[I-1]
            if J == 6: People[I][J] =maxPR[I-1]
            if J == 7: People[I][J] =avgPR[I-1]
            if J == 8: People[I][J]=sdPR[I-1]
            if J == 9: People[I][J]=kurtosisPR[I-1]

            if J == 10:  People[I][J] =minPosition[I-1]
            if J == 11:  People[I][J] =maxPosition[I-1]
            if J == 12:  People[I][J] =avgPosition[I-1]
            if J == 13:  People[I][J] =sdPosition[I-1]
            if J == 14:  People[I][J] =kurtosisPosition[I-1]

            if J == 15:  People[I][J] = minLight[I-1]
            if J == 16:  People[I][J] = maxLight[I-1]
            if J == 17:  People[I][J] = avgLight[I-1]
            if J == 18:  People[I][J] = sdLight[I-1]
            if J == 19:  People[I][J] = kurtosisLight[I-1]

            if J == 20:  People[I][J] = minOX_stat[I-1]
            if J == 21:  People[I][J] = maxOX_stat[I-1]
            if J == 22:  People[I][J] = avgOX_stat[I-1]
            if J == 23:  People[I][J] = sdOX_stat[I-1]
            if J == 24:  People[I][J] = kurtosisOX_stat[I-1]

            if J == 25:  People[I][J] = minAirflow[I-1]
            if J == 26:  People[I][J] = maxAirflow[I-1]
            if J == 27:  People[I][J] = avgAirflow[I-1]
            if J == 28:  People[I][J] = sdAirflow[I-1]
            if J == 29:  People[I][J] = kurtosisAirflow[I-1]

            if J == 30:  People[I][J] = minThorRes[I-1]
            if J == 31:  People[I][J] = maxThorRes[I-1]
            if J == 32:  People[I][J] = avgThorRes[I-1]
            if J == 33:  People[I][J] = sdThorRes[I-1]
            if J == 34:  People[I][J] = kurtosisThorRes[I-1]

            if J == 35:  People[I][J] =  minAbdoRes[I-1]
            if J == 36:  People[I][J] = maxAbdoRes[I-1]
            if J == 37:  People[I][J] = avgAbdoRes[I-1]
            if J == 38:  People[I][J] = sdAbdoRes[I-1]
            if J == 39:  People[I][J] = kurtosisAbdoRes[I-1]

            if J == 40:  People[I][J] = minEOG_L[I-1]
            if J == 41:  People[I][J] =  maxEOG_L[I-1]
            if J == 42:  People[I][J] = avgEOG_L[I-1]
            if J == 43:  People[I][J] = sdEOG_L[I-1]
            if J == 44:  People[I][J] = kurtosisEOG_L[I-1]

            if J == 45:  People[I][J] = minEOG_R[I-1]
            if J == 46:  People[I][J] = maxEOG_R[I-1]
            if J == 47:  People[I][J] = avgEOG_R[I-1]
            if J == 48:  People[I][J] = sdEOG_R[I-1]
            if J == 49:  People[I][J] =  kurtosisEOG_R[I-1]

            if J == 50:  People[I][J] = minEEG_sec[I-1]
            if J == 51:  People[I][J] = maxEEG_sec[I-1]
            if J == 52:  People[I][J] = avgEEG_sec[I-1]
            if J == 53:  People[I][J] = sdEEG_sec[I-1]
            if J == 54:  People[I][J] = kurtosisEEG_sec[I-1]

            if J == 55:  People[I][J] = minEEG[I-1]
            if J == 56:  People[I][J] = maxEEG[I-1]
            if J == 57:  People[I][J] = avgEEG[I-1]
            if J == 58:  People[I][J] = sdEEG[I-1]
            if J == 59:  People[I][J] = kurtosisEEG[I-1]

            if J == 60:  People[I][J] = minECG[I-1]
            if J == 61:  People[I][J] = maxECG[I-1]
            if J == 62:  People[I][J] = avgECG[I-1]
            if J == 63:  People[I][J] = sdECG[I-1]
            if J == 64:  People[I][J] = kurtosisECG[I-1]

            if J == 65:  People[I][J] = minEMG[I-1]
            if J == 66:  People[I][J] = maxEMG[I-1]
            if J == 67:  People[I][J] = avgEMG[I-1]
            if J == 68:  People[I][J] = sdEMG[I-1]
            if J == 69:  People[I][J] = kurtosisEMG[I-1]

    #print(People)
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

