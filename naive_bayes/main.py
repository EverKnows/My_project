# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:55:13 2018

@author: Ranly
"""
import numpy as np
import math
from sklearn import datasets, model_selection, naive_bayes


def loadData():
    postingList = [['my', 'dog', 'has', 'clever', 'problems', 'help', 'please'],
                   ['silly', 'fucking', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'poor', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def creatVocab(dataSet):
    vocaset = set([])
    for doc in dataSet:
        vocaset = vocaset | set(doc)
    return list(vocaset)

def to_standardForm(vocaList,inputset):
    m = len(vocaList)
    returnVec = np.zeros((m))
    for word in inputset:
        if word in vocaList:
            returnVec[vocaList.index(word)] = 1
        else: 
            print("not my word!")
    return returnVec

def trainNB(trainMatrix,trainCategory):
    
    num_Docs = len(trainMatrix)
    num_word = len(trainMatrix[0])
    p1_num = np.ones((1,num_word)); p1_deno = 2.0
    p0_num = np.ones((1,num_word)); p0_deno = 2.0  
    p_Abusive = np.sum(trainCategory,axis=0)/float(num_Docs)
    for i in range(num_Docs):
        if trainCategory[i] == 1:
            p1_num += trainMatrix[i]
            p1_deno += np.sum(trainMatrix[i],axis = 0)
        else:
            p0_num += trainMatrix[i]
            p0_deno += np.sum(trainMatrix[i],axis = 0)
    p0_w =p0_num/float(p0_deno)
    p1_w =p1_num/float(p1_deno) 
    return p0_w,p1_w,p_Abusive

def classify(inputdata,p0_w,p1_w,p_Abusive):
    
    result1 = ['nooooo!']
    result0 = ['yes!!!!']
    p0 = np.sum(inputdata*p0_w)+np.log(p_Abusive)
    p1 = np.sum(inputdata*p1_w)+np.log(1-p_Abusive)
    if p0>p1:
        return result0
    else:
        return result1
    
def main0():
    listofPost,label = loadData()
    vocaList = creatVocab(listofPost)
    trainMat = []
    for doc in listofPost:
        trainMat.append(to_standardForm(vocaList,doc))
    p0,p1,pc = trainNB(trainMat,label)
    testDoc = ['my','silly','dog','is','fucking','clever']
    testDoc0 = to_standardForm(vocaList,testDoc)
    print("%s is %s" %(testDoc,classify(testDoc0,p0,p1,pc)))
    
    
def load_sklearn_dataset(dataname = 'iris'):
    if dataname == 'iris':
        data = datasets.load_iris()
    elif dataname == 'wine':
        data = datasets.load_wine()
    elif dataname == 'boston':
        data = datasets.load_boston()
    elif dataname  == 'cancer':
        data = datasets.load_breast_cancer()
    elif dataname == 'digits':
        data = datasets.load_digits()
    else: 
        pass
    return model_selection.train_test_split(data.data,data.target,test_size = 0.15,random_state = 0,stratify = data.target)

def main1():
    train_X,test_X,train_y,test_y = load_sklearn_dataset('boston') 
    #data = datasets.load_boston()
    #train_X = data.data
    #train_y = data.target
    #print(np.shape(train_y))
    clf = naive_bayes.GaussianNB(priors = None)
    clf.partial_fit(train_X,train_y,np.unique(train_y))
  # pre_y = clf.predict(test_X)
    print(clf.scores(test_X,test_y))
main1()     
        
        
#https://blog.csdn.net/fontthrone/article/details/78824576