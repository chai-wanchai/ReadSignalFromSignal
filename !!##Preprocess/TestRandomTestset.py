
import pandas as pd

from sklearn.model_selection import train_test_split

import os
import datetime as T
s = T.datetime.now()
print("This program is split testset and trainset !!!")
testSize = (float)(input("Enter test size to split (0-1):"))
pathOpen = input("Enter path of file to split:")
file = [f for f in os.listdir(pathOpen) if f.endswith('.csv')]
for i in range(len(file)):
    print(i,file[i])
EnterFile = (int)(input("Select file number:"))
fullpath = pathOpen+"\\"+file[EnterFile]

df = pd.read_csv(fullpath)

train, test = train_test_split(df, test_size=testSize,random_state=2)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
test.to_csv(pathOpen+"\\"+"Testset_"+file[EnterFile],index=0)
train.to_csv(pathOpen+"\\"+"Trainset_"+file[EnterFile],index=0)
print(T.datetime.now()-s)
'''
train_data = data[:50]
test_data = data[50:]
print(np.asarray(train_data))
print(np.asarray(test_data))
print()'''