
import pandas as pd

from sklearn.model_selection import train_test_split

import os
import datetime as T
def TrainsetAndTestset(pathOpen):
    s = T.datetime.now()
    print("This program is split testset and trainset !!!")
    testSize = (float)(input("Enter test size to split (0-1):"))


    df = pd.read_csv(pathOpen)

    train, test = train_test_split(df, test_size=testSize,random_state=2)
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    test.to_csv(pathOpen+"_Testset.csv")
    train.to_csv(pathOpen+"_Trainset.csv")
    print(T.datetime.now()-s)
