import pyedflib as edf
import ListFile as F
import numpy as np
p = input("Enter path:")
file = F.lookFile(p, '.edf')
read = edf.EdfReader(file)

print("Sample Frequencies:",read.getSampleFrequencies())
print("Length Samples\n",read.getNSamples())
for i,label in enumerate(read.getSignalLabels()):
    print("({0})".format(label),np.asarray(read.readSignal(i,0,5)))
    print("\n")

#print(read.getSignalHeaders())