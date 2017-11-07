import pandas as pd
path ='C:\\Users\\chai_\\Google Drive\\1_2560 (1)\\308- Project2' \
      '\\edf\\shhs2\\Testset_SeparateEachEpochPerRow_Signal30s_Normalization.csv'
head = ['subject']
p = pd.read_csv(path,usecols=head).head(50)
p.to_csv('ii.csv')
print()

