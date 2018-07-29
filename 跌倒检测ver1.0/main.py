import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

yea = 1
no  = 0

dataSet = []

for df in pd.read_csv(
        "1-1.csv",
        encoding="gb2312",
        chunksize = 1,
        usecols = ["ACC_X","ACC_Y","ACC_Z"]
        ):
    dataSet.append(df)
    
m = len(dataSet)
acc_list = []

for i in range(115):
    acc_X = dataSet[i].get("ACC_X")
    acc_Y = dataSet[i].get("ACC_Y")
    acc_Z = dataSet[i].get("ACC_Z")
    acc_All = sqrt(acc_X**2+acc_Y**2+acc_Z**2)
    acc_list.append(float(acc_Z/100))

min_Index = acc_list.index(min(acc_list))
acc_all_list = []
acc_all_list.append(yea)

filename = 'test.txt'
def text_save(acc_all_list,filename,mode = 'a'):
    fr = open(filename,mode='a')
    for i in range(len(acc_all_list)):
        fr.write(str(acc_all_list[i])+' ')
        fr.flush()
    fr.write('\n')
    fr.flush()
    fr.close()
    
text_save(acc_all_list,filename)

t = linspace(0,0.04,115)
plt.figure(figsize=(8,4))
plt.plot(t,acc_list,label="$ACC$",color="red",linewidth=2)
plt.xlabel("TIME(S)")
plt.ylabel("ACC")
plt.xlim(0,0.045)
plt.ylim(-100,500)
plt.title("Acceleration over time chart")
plt.show()
    
    