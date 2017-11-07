import numpy as np
import pandas as pd
import os.path
import datetime as Time
import csv
from numba import jit

print("This script is tranpose Data in CSV and Delete String from column.")
a = Time.datetime.now()
path =input("Enter path of CSV file :")
file = [f for f in os.listdir(path) if f.endswith(".csv")]
for i in range(len(file)):
    print(i,file[i])
EnterFile = (int)(input("Enter number File :"))
print("open file: ",path+"\\"+file[EnterFile])
outPath =input("Enter path to save it :")
setOutputName = input("Enter file output name : ")+".csv"

@jit
def TranposeData():
    s=Time.datetime.now()
    Transpose = pd.DataFrame()
    df = pd.read_csv(path+"\\"+file[EnterFile], delimiter=",",dtype=object)
    Transpose = df.T
    Transpose.to_csv(outPath+"\\"+setOutputName,mode="a")
    e = Time.datetime.now()
    total =e-s
    return total


print(TranposeData())
