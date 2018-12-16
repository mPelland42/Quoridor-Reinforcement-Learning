# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:48:37 2018

@author: Sam
"""

import tensorflow as tf
#import os
import math
import os

LAYER_SIZE = 75
PROB_WIN_LAYER_SIZE_1 = 100
PROB_WIN_LAYER_SIZE_2 = 50

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

dir = os.path.dirname(os.path.realpath(__file__))+"\\"

class Model:
    def __init__(self, num_states, gridSize, num_actions, batch_size, restore, file):
        self._num_states = num_states
        self._num_actions = num_actions
        self._grid_size = gridSize
        self._batch_size = batch_size
        
        # define the placeholders
        self._states = None
        self._actions = None
        
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        
        
        # Predictd wins variables
        self.probWinInput = None
        self.predictWin = None
        self.ProbOptimizer = None
        
        # now setup the model
        self.defineCNN()
        self.file = file
        #self.defineModel()
        self.defineProbOfWinModel()
        
        self._var_init = tf.global_variables_initializer()
        
        self.saver = tf.train.Saver()
        
    def save(self, sess):
        local = self.saver.save(sess, "./te/"+self.file)
        print("saved to ", local)
        
    def load(self, sess):
        self.saver.restore(sess, "./te/"+self.file)
        
        
    def defineModel(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, LAYER_SIZE, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, LAYER_SIZE, activation=tf.nn.relu)
        
        self._logits = tf.layers.dense(fc2, self._num_actions)
        
        self.loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        
        
        
    def defineProbOfWinModel(self):
        self.probWinInput = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self.actualWin = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        self.fc1P = tf.layers.dense(self.probWinInput, PROB_WIN_LAYER_SIZE_1, activation=tf.nn.relu)
        self.fc2P = tf.layers.dense(self.fc1P, PROB_WIN_LAYER_SIZE_2, activation=tf.nn.relu)
        
        self.predictWin = tf.layers.dense(self.fc2P, 1)
        
        
        self.probLoss = tf.losses.mean_squared_error(self.actualWin, self.predictWin)
        self.probOptimizer = tf.train.AdamOptimizer().minimize(self.probLoss)

        
        
    def defineCNN(self):
        finalConvoSize = math.ceil(math.ceil((self._grid_size)/2)/2)
        weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,1,16])),
                   'W_conv2':tf.Variable(tf.random_normal([3,3,16,32])),
                   'W_fc':tf.Variable(tf.random_normal([finalConvoSize * finalConvoSize * 32,64])),
                   'out':tf.Variable(tf.random_normal([66, self._num_actions]))}
    
        biases = {'b_conv1':tf.Variable(tf.random_normal([16])),
                   'b_conv2':tf.Variable(tf.random_normal([32])),
                   'b_fc':tf.Variable(tf.random_normal([64])),
                   'out':tf.Variable(tf.random_normal([self._num_actions]))}
    
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        
        self.grid = tf.slice(self._states, [0, 0], [-1, self._num_states-2])
        self.grid = tf.reshape(self.grid, shape=[-1, self._num_states-2])
        
        self.wallsLeft = tf.slice(self._states, [0, self._num_states-2], [-1, 2])
        self.wallsLeft = tf.reshape(self.wallsLeft, shape = [-1, 2])
        
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
    
        x = tf.reshape(self.grid, shape=[-1, self._grid_size, self._grid_size, 1])
    
        conv1 = tf.nn.relu(self.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        self.conv1 = self.maxpool2d(conv1)
        
        conv2 = tf.nn.relu(self.conv2d(self.conv1, weights['W_conv2']) + biases['b_conv2'])
        self.conv2 = self.maxpool2d(conv2)
    
        fc = tf.reshape(self.conv2,[-1, finalConvoSize * finalConvoSize * 32])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)
        fc = tf.concat([fc, self.wallsLeft], 1)
    
        self._logits = tf.matmul(fc, weights['out'])+biases['out']
        
        
        
        
        self.loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        
        self._optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self._var_init = tf.global_variables_initializer()
        
    
    

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def maxpool2d(self, x):
        #                        size of window         movement of window
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    

        
    def getNumActions(self):
        return self._num_actions
    def getNumStates(self):
        return self._num_states
    def getBatchSize(self):
        return self._batch_size
        
    def predictOne(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self._num_states)})
            
    
    def predictBatch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})
    
    def trainBatch(self, sess, x_batch, y_batch):
        return sess.run([self._optimizer, self.loss], feed_dict={self._states: x_batch, self._q_s_a: y_batch})
    
            
    def predictOneProb(self, state, sess):
        return sess.run(self.predictWin, feed_dict={self.probWinInput: state.reshape(1, self._num_states)})
            
    

    
    def trainBatchProb(self, sess, x_batch, y_batch):
        
        #print(x_batch)
        #print(y_batch)
        #print(sess.run(self.fc1P, feed_dict={self.probWinInput: x_batch, self.actualWin: y_batch}))
        #print(sess.run(self.fc2P, feed_dict={self.probWinInput: x_batch, self.actualWin: y_batch}))
        #print(sess.run(self.predictWin, feed_dict={self.probWinInput: x_batch, self.actualWin: y_batch}))
        #print(sess.run(self.probLoss, feed_dict={self.probWinInput: x_batch, self.actualWin: y_batch}))
        
        return sess.run([self.probOptimizer, self.probLoss], feed_dict={self.probWinInput: x_batch, self.actualWin: y_batch})
    