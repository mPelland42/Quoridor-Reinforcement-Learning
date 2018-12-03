# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 00:35:45 2018

@author: Sam
"""
import tensorflow as tf

class DeepNetwork:
    
    def nn(self, inputs, layers_sizes, scope_name):
        """Creates a densely connected multi-layer neural network.
        inputs: the input tensor
        layers_sizes (list<int>): defines the number of units in each layer. The output 
            layer has the size layers_sizes[-1].
        """
        with tf.variable_scope(scope_name):
            for i, size in enumerate(layers_sizes):
                inputs = tf.layers.dense(
                    inputs,
                    size,
                    # Add relu activation only for internal layers.
                    activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name=scope_name + '_l' + str(i)
                )
        return inputs