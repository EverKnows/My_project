# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:34:23 2018
Copyright © Sat Sep 15 20:34:23 2018 by Ranly
@author: Ranly
@E-mail：1193932296@qq.com
        I PROMISE
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_train
import mnist_inference

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'
DATA_DIR = ROOT_DIR + 'fashion/'

tf.reset_default_graph()

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(name='input-x',shape=[None,784],dtype=tf.float32)
        y_ = tf.placeholder(name='output-y',shape=[None,10],dtype=tf.float32)
        
        y = mnist_inference.inference(x,None)
        
        average_variable = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)    
        saver = tf.train.Saver(average_variable.variables_to_restore())
        
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                acc = sess.run(acc,feed_dict=validate_feed)
                print('the final acc is {0}'.format(acc))
            else:
                print('the model is not exist!')

def main(argv=None):
    mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
                