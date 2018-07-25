from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():  ##loading DataSet
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

def sigmoid(inX):          
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatrix,classLabel):  ##gradAscent 
    dataMatrix = mat(dataMatrix)
    labelMatrix = mat(classLabel).transpose()
    numCycles = 500
    m,n = shape(dataMatrix)
    weights = ones((n,1))
    alpha = 0.001
    for i in range(numCycles):
        out = sigmoid(dataMatrix*weights)
        error = labelMatrix - out
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):   
    dataMat,labelsMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if (int(labelsMat[i]) == 1):
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0 ,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]  
    y = y.reshape(60,-1)
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent(dataMatrix,classLabel):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   
    for i in range(m):
        #out = dot(dataMatrix[i],weights)
        #h = sigmoid(out)
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabel[i] - h  
        
        dataMat = mat(dataMatrix[i]).transpose()
        
        #weights = weights + alpha*error*array(dataMat)    
        weights = weights + alpha*error*dataMatrix[i]   
    return weights
def stocGradAscent1(dataMatrix,classLabels,numiter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numiter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights +alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
    
def classVector(inX,weights):
    out = sigmoid(sum(inX*weights))
    if(out >= 0.5):
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        lineArr=[]
        currLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[-1]))
    trainingWeights = stocGradAscent1(array(trainingSet),trainingLabel,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        lineArr = []
        numTestVec +=1
        currLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if( int(classVector(array(lineArr),trainingWeights)) !=  int(currLine[-1]) ):
            errorCount+=1
    errorRate = float(errorCount)/numTestVec
    print "the errorrate is :%f"%errorRate
    return errorRate
            
def mutiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum +=colicTest()
    print "after %d iterations the average error rate is::%f"%(numTests,errorSum/float(numTests))
    
colicTest();
        




    
 
    
    