import  os
path = input("Enter path of File :")
file = [f for f in os.listdir(path) if f.endswith('.csv')]
for f in file:
    print(f)
select = (int)(input("Enter number of file :"))

