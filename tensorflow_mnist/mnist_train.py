# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:33:12 2018
Copyright © Sat Sep 15 14:33:12 2018 by Ranly
@author: Ranly
@E-mail：1193932296@qq.com
        I PROMISE
"""

import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data 
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'
DATA_DIR = ROOT_DIR + 'fashion/'
MODEL_SAVE_PATH = ROOT_DIR + 'model_save/'
MODEL_NAME = 'model.ckpt'
BATCH_SIZE = 100          
LEARNING_RATE_BASE = 0.8   
LEARNING_RATE_DECAY = 0.99
REGULARIZATON_RATE = 0.0001
TRAINING_STEPS = 300
MOVING_AVERAGE_DECAY = 0.99 

tf.reset_default_graph()

def train(mnist):
    x = tf.placeholder(dtype=tf.float32,shape=[None,784],\
                       name='input-x')
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,10],\
                        name='output-y')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATON_RATE)
    y = mnist_inference.inference(x,regularizer)
    
    global_step = tf.Variable(0,name='global_step',trainable=False)
    average_variable = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,\
                                                         global_step) 
    average_variable_op = average_variable.apply(tf.trainable_variables())
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,\
                                               global_step,BATCH_SIZE,LEARNING_RATE_DECAY)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,\
                                                                  labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('Loss'))
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    total_op = tf.group([train_op,average_variable_op])
    
    accurary = tf.reduce_mean(tf.cast(tf.equal( tf.argmax(y_,1),tf.argmax(y,1) ),dtype=tf.float32))
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init =  tf.initialize_all_variables()
        sess.run(init)
        for i in range(200):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
            _,acc = sess.run([total_op,accurary],feed_dict=validate_feed)
            print('the acc is {0}'.format(acc))
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

     
def main(argv=None):
    mnist = input_data.read_data_sets(DATA_DIR,one_hot = True)
    train(mnist)
if __name__ == '__main__':    
    tf.app.run()
            
    
    
    
    
    
    
    
    
    
    
    
    
    