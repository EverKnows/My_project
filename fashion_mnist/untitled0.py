# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:36:04 2018

@author: lenovo
"""
import matplotlib.pyplot as plt
import numpy as np
target = 0.5
x_ = np.arange(-10,10,0.01)
length =len(x_)
y_=[]
for i in range(length):
    y_.append(target);
r = 1
a = 0
b = 0
theta = np.arange(0, 2*np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)
fig = plt.figure() 
axes = fig.add_subplot(111) 
axes.plot(x, y)
axes.plot(x_,y_,'g')
axes.plot(sqrt(r*r-target**2),target,'r')
axes.axis('equal')



        
        

