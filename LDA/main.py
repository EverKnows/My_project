# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 19:09:12 2018

@author: Ranly
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

def loadData(filename):
    fr = open(filename)
    dataSet = []
    label = []
    for line in fr.readlines():
        line = line.strip().split()
        dataSet.append(line[:-1])
        label.append(line[-1])
    return dataSet,label

class LDA:
    def __init__(self,classnum = 2):
        self.classnum = 2
    
    def dataProcess(self,X,y):  #to divide two data
        dataSet1 = []
        label1 = []
        dataSet0 = []
        label0 = []
        m,n = np.shape(X)
        for i in range(m):
            if y[i] == 1:
                dataSet1.append(X[i])
                label1.append(y[i])
            else:
                dataSet0.append(X[i])
                label0.append(y[i])
        return dataSet0,label0,dataSet1,label1
    
    def means(self,dataSet):
        data = np.array(dataSet)
       # print(type(dataSet))
        mean_vector = np.mean(data,axis=0)
        mean_vector = np.mat(mean_vector)
      #  mean_vector = mean_vector.astype(float)
        return mean_vector
    
    def caulS_w(self,dataSet,m1):
        m = np.shape(dataSet)[1]
        S_w = np.mat(np.zeros((m,m)))
        S_w = S_w.astype(float)
        for i in range(m):
            u1 = np.mat(dataSet[i]-m1)
           # u1 = u1.astype(float)
            S_w = S_w + u1.T*u1
        return S_w
            
                
    def fit(self,X,y):
        dataSet0,label0,dataSet1,label1 = self.dataProcess(X,y)
        m0 = self.means(dataSet0)
        m1 = self.means(dataSet1)
        S_w_class1 = self.caulS_w(dataSet1,m1)
        S_w_class0 = self.caulS_w(dataSet0,m0)
        S_w = S_w_class1 + S_w_class0
        S_w1 = np.linalg.inv(S_w)
        W = S_w1*(m0-m1).T       
        return W,m0,m1
    
    def classfiy(self,X,y):
        W,m0,m1 = self.fit(X,y)
       # print(np.shape(W))
        m = np.shape(X)[0]
        predictX = []
        for i in range(m):
            if X[i]*W > 0.5*W.T*(m0+m1).T:
                predictX.append(1)
            else:
                predictX.append(0)
        return predictX
    
def main():
    # X,y = loadData('horseColicTraining.txt')
    dataSet = load_iris()
    X = dataSet.data
    y = dataSet.target 
    m = np.shape(X)[0]
    for i in range(m):
        if y[i] == 2:
            np.delete(X,i,axis = 0)
            np.delete(y,i,axis = 0)
    
    lda = LDA()
    predictX = lda.classfiy(X,y)
   # m = np.shape(X)[0]
    errorcount = 0
    for i in range(m):
        if predictX[i] != y[i]:
            errorcount+=1       
    print("the errorcount is: %s"%errorcount)
    print("the total is :%s"%m)
def sk_lda(solver='svd'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    dataSet = load_iris()
    X = dataSet.data
    y = dataSet.target 
    m = np.shape(X)[0]
    if solver == 'lsqr':
        errorcount = 0
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',n_components=2)
        lda.fit(X,y)
        y_pre = lda.predict(X)
        for i in range(m):
            if y_pre[i] != y[i]:
                errorcount+=1       
        print("the errorcount is: %s"%errorcount)
        print("the total is :%s"%m)
    if solver == 'svd':
        lda = LinearDiscriminantAnalysis(solver='svd',n_components=2)
        X_new = lda.fit(X,y).transform(X)
        plt.scatter(X_new[:,0],X_new[:,1],marker = 'o',c=y)
        plt.show()
    if solver == 'eigen':
        lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto',n_components=2)
        X_new = lda.fit(X,y).transform(X)
        plt.scatter(X_new[:,0],X_new[:,1],marker = 'o',c=y)
        plt.show()
        
if __name__ == '__main__':
    solver = ['lsqr','svd','eigen']
    for i in solver:
        sk_lda(i)
        

        