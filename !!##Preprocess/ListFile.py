import os

def lookFile(path,type):
    file = [f for f in os.listdir(path) if f.endswith(type)]
    for i,f in enumerate(file):
        print(i,f)
    select = (int)(input("Select file number:"))
    fullpath = path + "\\" + file[select]
    return  fullpath
