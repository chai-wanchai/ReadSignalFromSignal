import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from subprocess import call
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import os
import numpy as np
#import matplotlib as plt
import datetime as T
from sklearn import svm
#############  Global Variable ###################
path =""
file = ""
selectTrain = 0
selectTest = 0
fullpathTrain= ""
fullpathTest =""
SelectColumn = ['SaO2_min', 'SaO2_max', 'SaO2_avg', 'SaO2_SD', 'SaO2_kurtosis',
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

############  function Calulate Score Parameter as Accuracy Speccifity Sensivity ##################
def CalScoreParameter(confusion,label):
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.values.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F-meature
    F_measure = 2 * (PPV * TPR) / (PPV + TPR)
    # AUC
    AUC = auc(FPR, TPR,reorder=True)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print(ACC)
    text_file.write("\n---------------- {0} ------------------------\n".format(label))
    text_file.write("\n################ overall accuracy ###########\n {0}"
                    "\n#############################################".format(ACC))


    text_file.write("\n##################  TPR  ######################\n{0}"
                    "\n##################  TNR  ######################\n{1}"
                    "\n##################  PPV  ######################\n{2}"
                    "\n##################  NPV  ######################\n{3}"
                    "\n##################  FPR  ######################\n{4}"
                    "\n##################  FNR  ######################\n{5}"
                    "\n##################  FDR  ######################\n{6}"
                    "\n##################  AUC  ######################\n{7}"
                    "\n##################  F_measure  ######################\n{8}"
                    .format(TPR, TNR, PPV, NPV, FPR, FNR, FDR,AUC , F_measure))
    text_file.write("\n-------------------------------------------------\n")

###################################################################################################

##################  test data #####################
if __name__ == '__main__':
    s = T.datetime.now()
    print("################# This program is Decision Trees Model ################################")
    print("###################  Cross Validation  #####################")
    path = input("Enter path of Train File :")
    file = [f for f in os.listdir(path) if f.endswith('.csv')]
    for i in range(len(file)):
        print(i, file[i])

    selectTrain = (int)(input("Enter number of Train file :"))
    selectTest =(int)(input("Enter number of Test file :"))
    fullpathTrain = path+"\\"+file[selectTrain]
    fullpathTest = path+"\\"+file[selectTest]
    nameOfFile = file[selectTrain].replace('Trainset','').replace('SeparateEachEpochPerRow','SVM').replace(".csv", ".txt")
    text_file = open(path + "\\Output_" + nameOfFile, "w")
    text_file.write("Start Time: "+str(s))
    text_file.write("\nRead from file :" + file[selectTrain])
    text_file.write("\nRead from file :" + file[selectTest])


    df_Train = pd.read_csv(fullpathTrain)
    df_Test = pd.read_csv(fullpathTest)
    text_file.write("\nData Train:\n" + str(df_Train.head()))
    print('=================head')
    print(df_Train.head(5))
    #convert string

    df_Train=df_Train[SelectColumn]
    df_Train =df_Train.dropna()
    X_train = df_Train.drop('stage', axis=1)
    y_train = df_Train['stage']


    df_Test=df_Test[SelectColumn]
    df_Test =df_Test.dropna()
    X_test = df_Test.drop('stage', axis=1)
    y_test = df_Test['stage']
    print('=============================Tail')
    print(df_Train.tail())
    print('=============================y')
    print(df_Train['SaO2_min'].count())
    print("################### Crossvalidation  ##################")
    model = svm.SVC(kernel='sigmoid',probability=True,cache_size=3)
    ###############  Crossvalidation  ###################

    kf = KFold(n_splits=10)
    k=1
    #print(X_train.loc[0])
    for train_index, test_index in kf.split(X_train):
        model_Cross = model
        #print(train_index,test_index)
        X_train_Cross, X_test_Cross = X_train.loc[train_index], X_train.loc[test_index]
        y_train_Cross, y_test_Cross = y_train.loc[train_index], y_train.loc[test_index]
        X_train_Cross= X_train_Cross.dropna()
        X_test_Cross =X_test_Cross.dropna()
        y_train_Cross=y_train_Cross.dropna()
        y_test_Cross =y_test_Cross.dropna()
        #print("X_Train:",X_train_Cross)
        model_Cross.fit(X_train_Cross, y_train_Cross)
        y_predict_Cross = model_Cross.predict(X_test_Cross)
        #print(y_predict_Cross)
        confusion_matrix_cross = pd.DataFrame(confusion_matrix(y_test_Cross, y_predict_Cross))
        CalScoreParameter(confusion_matrix_cross,"K-"+str(k))
        k+=1

    scores = cross_val_score(model, X_train, y_train, cv=10)
    text_file.write("\n##############  K-Fold #####################\n")
    for i in range(len(scores)):
        text_file.write("\nK-{0} : {1}\n".format(i + 1, scores[i]))
    text_file.write("\nMean K- Flod : {0}".format(np.mean(scores)))
    print('==================accuracy crossvalidation ==============')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Score : \n {0}".format(scores))


    ######  Train set and Test set ########
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('==================accuracy normal =======================')
    print(accuracy_score(y_test, y_predict))
    accuracy_TESTTRAIN =accuracy_score(y_test, y_predict)

    #confusion matrix
    confusion=pd.DataFrame(confusion_matrix(y_test, y_predict))
    print(confusion)
    CalScoreParameter(confusion,"From Train and Test set")
    e = T.datetime.now()
    ######################3  Write Result to File #############################3

    '''
    text_file.write("\nFeature: \n")
    text_file.write(str(X))
    text_file.write("\n Target : \n {0}".format(y))
    '''

    text_file.write("\n############## Accuracy Train 70% and Test 30 % ##################\n "
                    "{0} \n##########################################\n".format(accuracy_TESTTRAIN))
    text_file.write("\n############## Confusion Matrix 70/30 ############ \n {0}"
                    .format(confusion))
    text_file.write("\nEnd Time: "+str(e))
    text_file.write("\nTotal Time : "+str(e-s))
    text_file.close()
    print(e-s)
    print("Output :",path)