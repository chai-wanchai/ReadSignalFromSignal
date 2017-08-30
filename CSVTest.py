import numpy as np
#epoch = ["epoch"]#+np.array(["555"])#+np.array([f+1 for f in range(10)])
#epoch.append('uuuu')
#FileName = ["EEG"]+epoch
#print(FileName)

for n in range(10):
	temp ="epoch"+str(n)
	hh=np.append(temp)
	'''epoch[n]=str(temp)
	print(epoch[n])
	'''

print(hh)