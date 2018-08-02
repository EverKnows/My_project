# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:43:56 2018

@author: Ranly
"""
import tensorflow as tf
import fashion_mnist_load
import os
from sklearn import preprocessing
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#data path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'
DATA_DIR = ROOT_DIR + 'fashion/'

#preprocessing，it is not necessary for the tensorflow
##preprocessing     
#labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
#x,y = fashion_mnist_load.load_mnist(DATA_DIR)
#images = preprocessing.scale(x)
#Labels = np.array([labels[j] for j in y])
#images_train = images[0:6000]
#labels_train = Labels[0:6000]

#images_validate = images[6000:7000]
#labels_validate = Labels[6000:7000]

#images_test = images[7000:8000]
#labels_test = Labels[7000:8000]


#initialize neuron net parameter
INPUT_NODE = 784           #input image size,fashion mnist's size is 28*28
OUTPUT_NODE = 10           #the num of class

LAYER1_NODE = 500          #layer1's node
BATCH_SIZE = 100           #batch'size, for the training

LEARNING_RATE_BASE = 0.8   
LEARNING_RATE_DECAY = 0.99
REGULARIZATON_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99  #for Moving average model
#to calulate the result of layer
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:  #if not provide with Moving average model's parameter,use the default parameter
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:    #this two layer net use reLu activation function
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(fashion_mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    #initialize the parameter of hidden layer
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #initialize the parameter of input layer
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y = inference(x,None,weights1,biases1,weights2,biases2)
    
    global_step = tf.Variable(0,trainable = False)
    #initialize the Moving average model
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #use Moving average model in all parameters
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #caulate  the  forward propagation result with moving average model
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #caculate the cross entropy 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #regularzation
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATON_RATE)
    regularization = regularizer(weights2) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,BATCH_SIZE,LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #start caculate
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:fashion_mnist.validation.images,y_:fashion_mnist.validation.labels}
        test_feed = {x:fashion_mnist.test.images,y_:fashion_mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict = validate_feed)
      #  index = [ i for i in range(6000)]
      #  random_index = []
      #  random_index = np.random.choice(random_index,BATCH_SIZE)
        
      #  images_train_batch =  np.array(images_train)[random_index]
      #  images_train_batch = images_train_batch.tolist()
        
      #  labels_train_batch = np.array(labels_train)[random_index]
      #  labels_train_batch = labels_train_batch.tolist()
            xs,ys = fashion_mnist.train.next_batch(BATCH_SIZE) 
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            test_acc = sess.run(accuracy,feed_dict = test_feed)
            print("After %d training step(s),test accuracy using average""model is %g "% (TRAINING_STEPS,test_acc))
def main(argv = None):  
    fashion_mnist = input_data.read_data_sets(DATA_DIR,one_hot = True) #loading the fashion mnist
    train(fashion_mnist)

if __name__ == "__main__":
    tf.app.run()
    



 






