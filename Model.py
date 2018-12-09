# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:48:37 2018

@author: Sam
"""

import tensorflow as tf
import os

LAYER_SIZE = 70

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        
        # define the placeholders
        self._states = None
        self._actions = None
        
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        
        # now setup the model
        self.defineModel()
        
        #self.dir = os.path.dirname(os.path.realpath(__file__))
        #self.modelSaver = tf.train.Saver(self._logits)
        

    def defineModel(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, LAYER_SIZE, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, LAYER_SIZE, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()
        
        
        
    #def updateModel(self):
    #    self.modelSaver.save(self.sess, 'test', global_step = 1)
        
        
    def getNumActions(self):
        return self._num_actions
    def getNumStates(self):
        return self._num_states
    def getBatchSize(self):
        return self._batch_size
        
    def predictOne(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self._num_states)})
    
    def predictBatch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})
    
    def trainBatch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})
        
        
    