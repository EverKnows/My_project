from numpy import *
import operator

class K_NN(object):
    def __init__(self,numofX,numofY):
        self.para = {}
        self.para['num_Feature'] = numofX
        self.para['num_Data'] = numofY

    def filetoMatrix(self,filename):
        fr = open(filename)
        arrayofLine = fr.readlines()
        returnMat = zeros((self.para['num_Data'],self.para['num_Feature']-1))
        index = 0
        Labels = []
        for line in arrayofLine:
            try:
                line = line.strip()
                listofLine = line.split(' ')
                returnMat[index,:] = listofLine[0:12]
                Labels.append(int(listofLine[-1]))
                index = index + 1
            except:
                pass
        return returnMat,Labels
    
    def autoNorm(self,dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals -minVals
        normDataSet = zeros(shape(dataSet))
        m = self.para['num_Feature']-1
        n = self.para['num_Data']
        normDataSet = (dataSet - tile(minVals,(n,1))) / tile(ranges,(n,1))
        return normDataSet,ranges,minVals
    
    def Normal_distribution(self,sqDiffMat,theta):
        m = sqDiffMat.shape[1]
        normsqDiffMat = zeros((shape(sqDiffMat)))
        norm_function = []
        mid = (m/2)-1
        
        for i in range(m):
            theta1 = theta**2
            index = (-(i-mid)**2)/(2*theta1) 
            norm_function.append(1/(sqrt(2*3.14)*theta)*exp(index))
            
        for i in range(m):
             sqDiffMat[i,:] = sqDiffMat[i,:]*norm_function 
        return sqDiffMat    
    
    def classify(self,normDataSet,inX,k,Labels,theta0):
        m = self.para['num_Feature']-1
        n = self.para['num_Data']
        diffMat = tile(inX,(n,1))-normDataSet
        sqDiffMat = diffMat**2
        NormalSqDiffMat = self.Normal_distribution(sqDiffMat,theta0)
        sqDistance = NormalSqDiffMat.sum(axis = 1)
        sqDistanceSort = sqDistance.argsort()
        classCount = {}
        for i in range(k):
            voteLabel = str(Labels[sqDistanceSort[i]])
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        sortClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
        return sortClassCount
    
total = 0
num_Right = 0
acc_Rate = []
labels = []
acc_list = []
filename = 'test.txt'
model = K_NN(13,13)
acc_list,labels = model.filetoMatrix(filename)
t = linspace(0.5,20,20)

for thetaX in t:
    for i in range(12):
        inx = acc_list[i]
        label = model.classify(acc_list,inx,5,labels,thetaX)
        total = total + 1
        if str(labels[i]) == label[0][0]:
            num_Right = num_Right + 1
        
    acc_Rate.append(float(num_Right)/total)
print 'the best theta is :%d,and the acc_Rate is :%f'%(t[acc_Rate.index(max(acc_Rate))],max(acc_Rate))
    
    