import dask.dataframe as dd
import datetime as T
import os
import pandas as pd
s = T.datetime.now()
path = input("Enter path of CSV file to read:")
file = [f for f in os.listdir(path) if f.endswith(".csv")]
for i in range(len(file)):
    print(i,file[i])
select = (int)(input("Select file number:"))
fullpath = path+"\\"+file[select]
df = pd.read_csv(fullpath)
print(df['epoch'].count())
print(df.head(df['epoch'].count()))
