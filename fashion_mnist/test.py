# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 17:22:42 2018

@author: lenovo
"""
import tensorflow as tf
import fashion_mnist_load
import os
from sklearn import preprocessing
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'
DATA_DIR = ROOT_DIR + 'fashion'

##preprocessing
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
x,y = fashion_mnist_load.load_mnist(DATA_DIR)
images = preprocessing.scale(x)

print(images)