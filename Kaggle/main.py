# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:08:23 2018
Copyright © Mon Sep  3 13:08:23 2018 by Ranly
@author: Ranly
@E-mail：1193932296@qq.com
        I PROMISE
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

train_path = 'C:/Users/lenovo/Desktop/train.csv'
test_path  = 'C:/Users/lenovo/Desktop/test.csv'
test_result_path = 'C:/Users/lenovo/Desktop/gender_submission.csv'

train_ = pd.read_csv(train_path)
test_ = pd.read_csv(test_path)
test_sur = pd.read_csv(test_result_path)

train_1 = train_.loc[:,['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
test_1 = test_.loc[:,['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


#fill the column age
median_age = train_1['Age'].median()
train_1.loc[train_1['Age'].isnull(),'Age'] = median_age
test_1.loc[test_1['Age'].isnull(),'Age'] = median_age

#fill the column embarked
word_counts = Counter(train_1['Embarked'])
top_1_word = word_counts.most_common(1)
train_1.loc[train_1['Embarked'].isnull(),'Embarked'] = top_1_word[0][0]
test_1.loc[test_1['Embarked'].isnull(),'Embarked'] = top_1_word[0][0]

#fill the column fare
fare_counts = Counter(test_1['Fare'])
fare = fare_counts.most_common(1)
test_1.loc[test_1['Fare'].isnull(),'Fare'] = fare[0][0]

#object to int or float
train_1['Sex'] = train_1['Sex'].map({'female':0,'male':1}).astype(int)
test_1['Sex'] = test_1['Sex'].map({'female':0,'male':1}).astype(int)

train_1['Embarked'] = train_1['Embarked'].map({'Q':1,'S':2,'C':3}).astype(int)
test_1['Embarked'] = test_1['Embarked'].map({'Q':1,'S':2,'C':3}).astype(int)

#use 'SibSp' and 'Parch' to generate a new feature 'isalone'
train_2 = train_1.loc[:,['PassengerId','Survived','Pclass','Sex','Age','Fare','Embarked']]
test_2 = test_1.loc[:,['PassengerId','Pclass','Sex','Age','Fare','Embarked']]

train_2['isalone'] = train_1['SibSp'] + train_1['Parch'] + 1
train_2['famliysize'] = train_2['isalone']
train_2.loc[train_2['isalone'] != 1,'isalone'] = 0

test_2['isalone'] = test_1['SibSp'] + test_1['Parch'] + 1 
test_2['famliysize'] = test_2['isalone']
test_2.loc[test_2['isalone'] != 1,'isalone'] = 0


#plot   isalone and sur_rate
isalone_sur = train_2[['isalone','Survived']].groupby('isalone').mean()
plt.subplot(331)
#plt.figure(figsize=(4,2))
plt.bar([0,1],[isalone_sur.loc[0,'Survived'],isalone_sur.loc[1,'Survived']],color = 'g',alpha = 0.5)
plt.ylim(0,1)
plt.xticks([0,1],['notalone','alone'])
plt.ylabel('sur_rate')
plt.text(0,isalone_sur.loc[0,'Survived']+0.1,str(round(isalone_sur.loc[0,'Survived'],3)),ha ='center')
plt.text(1,isalone_sur.loc[1,'Survived']+0.1,str(round(isalone_sur.loc[1,'Survived'],3)),ha ='center')
plt.title('isalone_sur')

#plot Pclass and sur_rate
Pclass_sur = train_2[['Pclass','Survived']].groupby('Pclass').mean()
plt.subplot(333)
#plt.figure(figsize=(4,2))
plt.bar([1,2,3],[Pclass_sur.loc[1,'Survived'],Pclass_sur.loc[2,'Survived'],Pclass_sur.loc[3,'Survived']],color = 'g',alpha = 0.5)
plt.ylabel('sur_rate')
plt.ylim(0,1)
plt.xticks(Pclass_sur.index,['no1','no2','no3'])
for x,y in zip(Pclass_sur.index,Pclass_sur.loc[:,'Survived']):
    plt.text(x,y+0.1,str(round(y,3)), ha = 'center')
plt.title('Pclass_sur')

#plot sex and sur_rate
Sex_sur = train_2[['Sex','Survived']].groupby('Sex').mean()
plt.subplot(337)
plt.bar(Sex_sur.index,Sex_sur.loc[:,'Survived'],color = 'g',alpha = 0.7)
plt.ylabel('sur_rate')
plt.ylim(0,1)
plt.xticks(Sex_sur.index,['female','male'])
for x,y in zip(Sex_sur.index,Sex_sur.loc[:,'Survived']):
    plt.text(x,y+0.1,str(round(y,3)), ha = 'center')
plt.title('Sex_sur')
plt.show()

"""
start training our machinelearing model
we plan to take the SVM to solve the problem

"""
train_X = train_2.loc[:,['Pclass','famliysize','Sex']].values
train_y = train_2.loc[:,['Survived']].values.ravel()

test_X = test_2.loc[:,['Pclass','famliysize','Sex']].values
test_y = test_sur.loc[:,['Survived']].values.ravel()

kernels = ['linear','poly','rbf','sigmoid']
C_list = [0.1,0.3,0.5,0.7,0.9,1.0]

best_score = 0.0

for Kernel in kernels:
    clf = svm.SVC(kernel = Kernel,C = 1.0)
    clf.fit(train_X,train_y)
#    print('the kernel {0}\'s correct_rate is {1} '.format(Kernel,clf.score(train_X,train_y)))
    if clf.score(train_X,train_y)>best_score:
        best_kernel = Kernel
        best_score = clf.score(train_X,train_y)

#use cross_validation set to choose C
best_C = 0.0
best_score = 0.0
train_X_,cv_X,train_y_,cv_y = train_test_split(train_X,train_y,test_size=0.3,random_state = 50 )
for c in C_list:
    clf = svm.SVC(kernel = 'rbf',C=c)
    clf.fit(train_X_,train_y_)
#    print('c:{0} and score{1}'.format(c,clf.score(cv_X,cv_y)))
    if clf.score(cv_X,cv_y) > best_score:
        best_score = clf.score(cv_X,cv_y)
        best_C = c
        
    
    
    
clf = svm.SVC(kernel=best_kernel,C=best_C)
clf.fit(train_X,train_y)
print('The test \'s correct_rate is {0}'.format(clf.score(test_X,test_y)))
print(' kernel:{0} ; C:{1}'.format(best_kernel,best_C))









