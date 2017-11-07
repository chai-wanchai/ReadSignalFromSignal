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
from sklearn.linear_model import SGDClassifier
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
if __name__ == '__main__':
    df_Train = pd.read_csv('C:\\Users\chai_\\Google Drive\\1_2560 (1)\\308- Project2\\edf\\shhs2\\Trainset_SeparateEachEpochPerRow_Signal30s_Normalization.csv')
    df_Test =pd.read_csv('C:\\Users\chai_\\Google Drive\\1_2560 (1)\\308- Project2\\edf\\shhs2\\Testset_SeparateEachEpochPerRow_Signal30s_Normalization.csv')
    df_Train = df_Train[SelectColumn]
    df_Train = df_Train.dropna()
    #print("Before:{0}".format(df_Train))

    X_train = df_Train.drop('stage', axis=1)
    y_train = df_Train['stage'].replace(2,1).replace(9,0).replace(3,1).replace(4,1).replace(5,2)

    df_Test = df_Test[SelectColumn]
    df_Test = df_Test.dropna()
    X_test = df_Test.drop('stage', axis=1)
    y_test = df_Test['stage'].replace(2,1).replace(9,0).replace(3,1).replace(4,1).replace(5,2)
    zero=0
    one=0
    two=0
    three=0
    four=0
    five=0
    six=0
    other=0
    for f in y_train:
        if f==0:zero=zero+1
        elif f==1:one=one+1
        elif f==2:two=two+1
        elif f==3:three=three+1
        elif f==4:four=four+1
        elif f==5:five=five+1
        elif f==6:six=six+1
        else:other=other+1

    print(max(y_train))
    print(min(y_train))
    print(zero,one,two,three,four,five,six,other)
    mins = df_Train.count()
    acti = ['identity', 'logistic', 'tanh', 'relu']
    sov =['lbfgs', 'sgd', 'adam']
    learn =['constant', 'invscaling', 'adaptive']
    for a in range(50,600,50):

                modelTree_1 = ExtraTreesClassifier(n_estimators=a, max_depth=None, criterion='entropy', n_jobs=-1)
                # modelTree_2 = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                # modelTree_3 = RandomForestClassifier(n_estimators=200, criterion='entropy',class_weight='balanced', n_jobs=-1)
                # modelTree_4 = RandomForestClassifier(n_estimators=200, criterion='entropy',oob_score=True,n_jobs=-1)
                #
                modelTree_1.fit(X_train, y_train)
                # modelTree_2.fit(X_train, y_train)
                # modelTree_3.fit(X_train, y_train)
                # modelTree_4.fit(X_train, y_train)
                y_predict1 = modelTree_1.predict(X_test)
                # y_predict2 = modelTree_2.predict(X_test)
                # y_predict3 = modelTree_3.predict(X_test)
                # y_predict4 = modelTree_4.predict(X_test)
                #
                accuracy1 = accuracy_score(y_test, y_predict1)
                # accuracy2 = accuracy_score(y_test, y_predict2)
                # accuracy3 = accuracy_score(y_test, y_predict3)
                # accuracy4 = accuracy_score(y_test, y_predict4)
                #
                print(a,"acc1:",accuracy1)
        # print("acc2:", accuracy2)
        # print("acc3:", accuracy3)
        # print("acc4:", accuracy4)
        # acc =[]
        # wei = ['distance','uniform']
        # algo =['auto','ball_tree','kd_tree','brute']
        #
        #
        # loss =  ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron','squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']
        # pen =['none', 'l2', 'l1', 'elasticnet']
        # for l in loss:
        #     for r in pen:
        #         modelTree = SGDClassifier(loss=l,penalty=r,max_iter=1000,n_jobs=-1)
        #         modelTree.fit(X_train, y_train)
        #         y_predict = modelTree.predict(X_test)
        #         accuracy = accuracy_score(y_test, y_predict)
        #         acc.append(accuracy)
        #         print(r,l,accuracy)
        # print(max(acc))
        # for index,u in enumerate(acc):
        #     if u == max(acc):
        #         print(index,u)

