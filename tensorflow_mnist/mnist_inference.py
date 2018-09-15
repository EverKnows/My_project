# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:20:22 2018
Copyright © Sat Sep 15 14:20:22 2018 by Ranly
@author: Ranly
@E-mail：1193932296@qq.com
        I PROMISE
"""

import tensorflow as tf

INPUT_NODE = 784           
OUTPUT_NODE = 10          
LAYER1_NODE = 500 

def inference(input_tensor,regularizer):
    with tf.variable_scope('Layer1',reuse= tf.AUTO_REUSE):
        bias = tf.get_variable(name='bias',shape=[LAYER1_NODE],\
                             initializer=tf.constant_initializer(0.1),dtype=tf.float32)
        weights = tf.get_variable(initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1),\
                         shape=[INPUT_NODE,LAYER1_NODE],name='Weight',dtype=tf.float32)
        if regularizer!=None:
            tf.add_to_collection('Loss',regularizer(weights))
        for variable in tf.trainable_variables('layer1'):
            print(variable.name)
        Layer1 = tf.nn.relu( tf.matmul(input_tensor,weights) + bias )
        
    with tf.variable_scope('Layer2',reuse= tf.AUTO_REUSE):
        bias = tf.get_variable(name='bias',shape=[OUTPUT_NODE],\
                             initializer=tf.constant_initializer(0.1),dtype=tf.float32)
        weights = tf.get_variable(initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1),\
                         shape=[LAYER1_NODE,OUTPUT_NODE],name='Weight',dtype=tf.float32)
        if regularizer!=None:
            tf.add_to_collection('Loss',regularizer(weights))
        Layer2 = tf.matmul(Layer1,weights) + bias
        for variable in tf.trainable_variables('layer2'):
            print(variable.name)
    return Layer2
        
        
        
        
        
        