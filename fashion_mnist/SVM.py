# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 08:13:37 2018

@author: Ranly
@ name : SVM_in_fashion_mnist
"""

import fashion_mnist_load
from sklearn import svm
from sklearn import preprocessing
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'
DATA_DIR = ROOT_DIR + 'fashion/'

images,labels = fashion_mnist_load.load_mnist(DATA_DIR);
images = images[0:2000]
labels = labels[0:2000]

images = preprocessing.scale(images)

model_svm = svm.SVC()

model_svm.fit(images,labels)

result = model_svm.predict(images)
xsize = images.shape[0]
num_of_right = (np.sum(labels == result))/xsize
#score = model_svm.score(images,labels)
print(num_of_right)


