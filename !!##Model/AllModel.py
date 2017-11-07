import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
import os
import numpy as np
import datetime as T
####################  Model Import  ######################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import xlsxwriter
#############  Global Variable ###################
workbook = xlsxwriter
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
    confusion = confusion.fillna(0)
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
    print(label,'\n',ACC)
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

    ###################----------------  Excel -----------------################

    worksheet.write(1, k, label, colFormat)
    parameter = ['ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'F_measure']
    # paraData = [ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR,F_measure]

    stage = len(ACC)

    i = 2
    columnK = k

    for para in parameter:

        worksheet.write(i, 0, para, redBold)
        ii = i
        for s in range(stage):
            ii += 1
            worksheet.write(ii, 0, 'stage' + str(s + 1))
            if para == 'ACC':
                try:
                    worksheet.write(ii, columnK, ACC[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')

            if para == 'TPR':
                try:
                    worksheet.write(ii, columnK, TPR[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'TNR':
                try:
                    worksheet.write(ii, columnK, TNR[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'PPV':
                try:
                    worksheet.write(ii, columnK, PPV[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'NPV':
                try:
                    worksheet.write(ii, columnK, NPV[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'FPR':
                try:
                    worksheet.write(ii, columnK, FPR[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'FNR':
                try:
                    worksheet.write(ii, columnK, FNR[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'FDR':
                try:
                    worksheet.write(ii, columnK, FDR[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
            if para == 'F_measure':
                try:
                    worksheet.write(ii, columnK, F_measure[s])
                except IndexError:
                    worksheet.write(ii, columnK, 'NULL')
        i = i + stage + 2
    worksheet.write(i, 0, 'AUC', redBold)
    worksheet.write(i, columnK, AUC)
    worksheet.write(i + 1, 0, 'Accuracy', redBold)
    worksheet.write(i + 1, columnK, accuracy)
    worksheet.write(i + 2, 0, 'Record')
    worksheet.write(i + 3, 0, 'Record Train')
    worksheet.write(i + 3, columnK, recordTrain)
    worksheet.write(i + 4, 0, 'Record Test')
    worksheet.write(i + 4, columnK, recordTest)

    ############################################################################

###################################################################################################
def Voting():
    print("---------------Model for Voting---------------")
    Vote = []
    NumberOfModel = (int)(input("Total Model for voting : "))

    for i in range(NumberOfModel):
        ii = (int)(input("Model {0}:".format(i+1)))
        Vote.append([ModelVote[ii],SelectModel(ii)])

    modelVoting = VotingClassifier(estimators=Vote,voting='hard')
    return  modelVoting

def SelectModel(NumberModel):

    if ModelVote[NumberModel] == 'Discision Tree':
        SelectModelVote.append('DT')
        model = DecisionTreeClassifier(class_weight='balanced',criterion='entropy',presort=True,random_state=10)
    if ModelVote[NumberModel] == 'Random Forest':
        SelectModelVote.append('RF')
        model = RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=-1)
    if ModelVote[NumberModel] == 'KNeighbors':
        SelectModelVote.append('KNN')
        model = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='auto', n_jobs=-1)
    if ModelVote[NumberModel] == 'Neuron Network':
        SelectModelVote.append('NN')
        model = MLPClassifier(hidden_layer_sizes=200,activation='logistic', solver='adam', alpha=1e-5, learning_rate='adaptive',
                              max_iter=10000)
    if ModelVote[NumberModel] == 'NaiveBayes':
        SelectModelVote.append('NB')
        model = GaussianNB()
    if ModelVote[NumberModel] == 'Extra Tree':
        SelectModelVote.append('ET')
        model = ExtraTreesClassifier(n_estimators=100, max_depth=None, criterion='entropy', n_jobs=-1)
    if ModelVote[NumberModel] == 'SVM':
        SelectModelVote.append('SVM')
        model = svm.SVC(kernel='sigmoid', probability=True)
    if ModelVote[NumberModel] == 'Voting':
        model = Voting()
    return model
##################  test data #####################
if __name__ == '__main__':
    s = T.datetime.now()
    print("################# This program is Voting Classify ################################")

    ModelVote = ['Discision Tree', 'Random Forest', 'KNeighbors', 'Neuron Network','NaiveBayes','Extra Tree','SVM','Voting']
    SelectModelVote = []
    print("Model for Test")
    for i, modelList in enumerate(ModelVote):
        print(i, " ", modelList)

    j = (int)(input("Select number of Model:"))
    model=SelectModel(j)


    ##############   Model  #####################


    #print(model.get_params())
    path = input("Enter path of Train File :")
    file = [f for f in os.listdir(path) if f.endswith('.csv')]
    for i in range(len(file)):
        print(i, file[i])

    selectTrain = (int)(input("Enter number of Train file :"))
    selectTest =(int)(input("Enter number of Test file :"))
    fullpathTrain = path+"\\"+file[selectTrain]
    fullpathTest = path+"\\"+file[selectTest]


    nameOfFile = file[selectTrain].replace('_','').replace('Trainset','').replace('SeparateEachEpochPerRow','').replace('Normalization',str([f for f in SelectModelVote])).replace(".csv", ".txt")

    text_file = open(path + "\\" + nameOfFile, "w")
    text_file.write("Start Time: "+str(s))
    text_file.write("\nRead from file :" + file[selectTrain])
    text_file.write("\nRead from file :" + file[selectTest])
    text_file.write("\nParameter:\n{0}\n{1}".format(str(model.get_params()),str(model)))

    if ModelVote[j]=='Voting':
        workbook = xlsxwriter.Workbook(path + "\\" + file[selectTrain].replace('Trainset_', 'Result').replace('SeparateEachEpochPerRow','').replace(".csv", ".xlsx").replace('Normalization',ModelVote[j]+str(SelectModelVote)), {'nan_inf_to_errors': True})
    else:
        workbook = xlsxwriter.Workbook(path + "\\" + file[selectTrain].replace('Trainset_', 'Result').replace('SeparateEachEpochPerRow','').replace(".csv", ".xlsx").replace('Normalization', ModelVote[j]), {'nan_inf_to_errors': True})

    nameOfSheet =file[selectTrain].replace('Trainset','').replace('SeparateEachEpochPerRow','').replace('Normalization','').replace(".csv", "").replace('_','')
    print(nameOfSheet)
    worksheet = workbook.add_worksheet(name=str(nameOfSheet))
    worksheet.set_column(0, 0, width=15.75)
    worksheet.set_column('B:L', width=12)
    headerFormat = workbook.add_format({'align': 'center', 'bold': True, 'font_color': 'red', 'font_size': 16, 'font_name': 'Tahoma'})
    colFormat = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#FFFF00', 'font_size': 11})
    redBold = workbook.add_format({'bold': True, 'font_color': 'red'})
    center = workbook.add_format({'align': 'center'})
    worksheet.write('A1', 'Epoch '+nameOfSheet.replace('Signal',''), headerFormat)
    if ModelVote[j] =='Voting':
        worksheet.write('B1', ModelVote[j], headerFormat)
        worksheet.write('C1',str(SelectModelVote),headerFormat)
    else:
        worksheet.write('B1', ModelVote[j], headerFormat)
    worksheet.write('E1', fullpathTrain)


    df_Train = pd.read_csv(fullpathTrain,dtype={
                'subject':str,'epoch':np.int8,
              'SaO2_min':np.float16, 'SaO2_max':np.float16, 'SaO2_avg':np.float16, 'SaO2_SD':np.float16, 'SaO2_kurtosis':np.float16,
              'PR_min':np.float16, 'PR_max':np.float16, 'PR_avg':np.float16, 'PR_SD':np.float16, 'PR_kurtosis':np.float16,
              'position_min':np.float16, 'position_max':np.float16, 'position_avg':np.float16, 'position_SD':np.float16, 'position_kurtosis':np.float16,
              'light_min':np.float16, 'light_max':np.float16, 'light_avg':np.float16, 'light_SD':np.float16, 'light_kurtosis':np.float16,
              'ox_stat_min':np.float16, 'ox_stat_max':np.float16, 'ox_stat_avg':np.float16, 'ox_stat_SD':np.float16, 'ox_stat_kurtosis':np.float16,
              'airflow_min':np.float16, 'airflow_max':np.float16, 'airflow_avg':np.float16, 'airflow_SD':np.float16, 'airflow_kurtosis':np.float16,
              'ThorRes_min':np.float16, 'ThorRes_max':np.float16, 'ThorRes_avg':np.float16, 'ThorRes_SD':np.float16, 'ThorRes_kurtosis':np.float16,
              'AbdoRes_min':np.float16, 'AbdoRes_max':np.float16, 'AbdoRes_avg':np.float16, 'AbdoRes_SD':np.float16, 'AbdoRes_kurtosis':np.float16,
              'EOG(L)_min':np.float16, 'EOG(L)_max':np.float16, 'EOG(L)_avg':np.float16, 'EOG(L)_SD':np.float16, 'EOG(L)_kurtosis':np.float16,
              'EOG(R)_min':np.float16, 'EOG(R)_max':np.float16, 'EOG(R)_avg':np.float16, 'EOG(R)_SD':np.float16, 'EOG(R)_kurtosis':np.float16,
              'EEG(sec)_min':np.float16, 'EEG(sec)_max':np.float16, 'EEG(sec)_avg':np.float16, 'EEG(sec)_SD':np.float16, 'EEG(sec)_kurtosis':np.float16,
              'EEG_min':np.float16, 'EEG_max':np.float16, 'EEG_avg':np.float16, 'EEG_SD':np.float16, 'EEG_kurtosis':np.float16,
              'ECG_min':np.float16, 'ECG_max':np.float16, 'ECG_avg':np.float16, 'ECG_SD':np.float16, 'ECG_kurtosis':np.float16,
              'EMG_min':np.float16, 'EMG_max':np.float16, 'EMG_avg':np.float16, 'EMG_SD':np.float16, 'EMG_kurtosis':np.float16,
              'stage':np.uint8})
    df_Test = pd.read_csv(fullpathTest,dtype={
                'subject':str,'epoch':np.int8,
              'SaO2_min':np.float16, 'SaO2_max':np.float16, 'SaO2_avg':np.float16, 'SaO2_SD':np.float16, 'SaO2_kurtosis':np.float16,
              'PR_min':np.float16, 'PR_max':np.float16, 'PR_avg':np.float16, 'PR_SD':np.float16, 'PR_kurtosis':np.float16,
              'position_min':np.float16, 'position_max':np.float16, 'position_avg':np.float16, 'position_SD':np.float16, 'position_kurtosis':np.float16,
              'light_min':np.float16, 'light_max':np.float16, 'light_avg':np.float16, 'light_SD':np.float16, 'light_kurtosis':np.float16,
              'ox_stat_min':np.float16, 'ox_stat_max':np.float16, 'ox_stat_avg':np.float16, 'ox_stat_SD':np.float16, 'ox_stat_kurtosis':np.float16,
              'airflow_min':np.float16, 'airflow_max':np.float16, 'airflow_avg':np.float16, 'airflow_SD':np.float16, 'airflow_kurtosis':np.float16,
              'ThorRes_min':np.float16, 'ThorRes_max':np.float16, 'ThorRes_avg':np.float16, 'ThorRes_SD':np.float16, 'ThorRes_kurtosis':np.float16,
              'AbdoRes_min':np.float16, 'AbdoRes_max':np.float16, 'AbdoRes_avg':np.float16, 'AbdoRes_SD':np.float16, 'AbdoRes_kurtosis':np.float16,
              'EOG(L)_min':np.float16, 'EOG(L)_max':np.float16, 'EOG(L)_avg':np.float16, 'EOG(L)_SD':np.float16, 'EOG(L)_kurtosis':np.float16,
              'EOG(R)_min':np.float16, 'EOG(R)_max':np.float16, 'EOG(R)_avg':np.float16, 'EOG(R)_SD':np.float16, 'EOG(R)_kurtosis':np.float16,
              'EEG(sec)_min':np.float16, 'EEG(sec)_max':np.float16, 'EEG(sec)_avg':np.float16, 'EEG(sec)_SD':np.float16, 'EEG(sec)_kurtosis':np.float16,
              'EEG_min':np.float16, 'EEG_max':np.float16, 'EEG_avg':np.float16, 'EEG_SD':np.float16, 'EEG_kurtosis':np.float16,
              'ECG_min':np.float16, 'ECG_max':np.float16, 'ECG_avg':np.float16, 'ECG_SD':np.float16, 'ECG_kurtosis':np.float16,
              'EMG_min':np.float16, 'EMG_max':np.float16, 'EMG_avg':np.float16, 'EMG_SD':np.float16, 'EMG_kurtosis':np.float16,
              'stage':np.uint8})

    print('=================head')



    df_Train=df_Train[SelectColumn]
    df_Train =df_Train.dropna()
    X_train = df_Train.drop('stage', axis=1)
    y_train = df_Train['stage']
    print(df_Train.head(5))
    text_file.write("\n{0}".format(df_Train.head(5)))

    df_Test=df_Test[SelectColumn]
    df_Test =df_Test.dropna()
    X_test = df_Test.drop('stage', axis=1)
    y_test = df_Test['stage']

    print('=============================Tail')
    print(df_Train.tail())
    print('=============================y')
    row = df_Train['stage'].count()
    print(row)
    text_file.write("\nTotal Row : "+str(row))

    ###############  Crossvalidation  ###################

    kf = KFold(n_splits=10)
    k=1
    recordKF = []
    scoresKFold =[]
    print("###################  Cross Validation  #####################")
    for train_index, test_index in kf.split(X_train):
        model_Cross = model

        X_train_Cross, X_test_Cross = X_train.loc[train_index], X_train.loc[test_index]
        y_train_Cross, y_test_Cross = y_train.loc[train_index], y_train.loc[test_index]
        X_train_Cross= X_train_Cross.dropna()
        X_test_Cross =X_test_Cross.dropna()
        y_train_Cross=y_train_Cross.dropna()
        y_test_Cross =y_test_Cross.dropna()
        #print("X_Train:",X_train_Cross)
        model_Cross.fit(X_train_Cross, y_train_Cross)
        y_predict_Cross = model_Cross.predict(X_test_Cross)
        accuracy = accuracy_score(y_test_Cross, y_predict_Cross)
        scoresKFold.append(accuracy)
        recordTest = y_test_Cross.count()
        recordTrain = y_train_Cross.count()
        confusion_matrix_cross = pd.DataFrame(confusion_matrix(y_test_Cross, y_predict_Cross))
        CalScoreParameter(confusion_matrix_cross,"K-"+str(k))
        recordKF.append(["Record K-{0}: Train({1}) Test({2})".format(k,recordTrain,recordTest)])

        k+=1


    text_file.write("\n##############  K-Fold #####################\n")

    for i in range(len(scoresKFold)):
        text_file.write("\nK-{0} : {1}\n\n{2}".format(i + 1, scoresKFold[i],recordKF[i]))
    text_file.write("\nMean K- Flod : {0}".format(np.mean(scoresKFold)))
    print('==================accuracy crossvalidation ==============')
    print("score k fold :", scoresKFold)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scoresKFold), np.std(scoresKFold) * 2))


    ######  Train set and Test set ########
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('==================accuracy Testset =======================')

    accuracy =accuracy_score(y_test, y_predict)
    print(accuracy)
    recordTrain=y_train.count()
    recordTest=y_test.count()
    confusion=pd.DataFrame(confusion_matrix(y_test, y_predict))
    print(confusion)
    CalScoreParameter(confusion,"Testset")
    e = T.datetime.now()
    ######################3  Write Result to File #############################3

    text_file.write("\nTrainset : {0}\nTestset :{1}".format(recordTrain,recordTest))
    text_file.write("\n############## Accuracy Train 70% and Test 30 % ##################\n "
                    "{0} \n##########################################\n".format(accuracy))
    text_file.write("\n############## Confusion Matrix 70/30 ############ \n {0}"
                    .format(confusion))
    text_file.write("\nEnd Time: "+str(e))
    text_file.write("\nTotal Time : "+str(e-s))
    text_file.close()
    workbook.close()
    print(e-s)
    print("Output :",path+"\\"+nameOfFile)
