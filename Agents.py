# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:54:50 2018

@author: Sam
"""

#from Game import Qoridor
from AgentState import AgentState
from DeepNetwork import DeepNetwork
import numpy as np

import tensorflow as tf
class Action:
    PAWN = "PAWN"
    WALL = "WALL"
    def __init__(self, actionType, X, Y):
            self.actionType = actionType
            self.X = X
            self.Y = Y
    def getType(self):
        return self.actionType
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    
    def __str__(self):
        return "TYPE: " + self.actionType +"  X,Y: " + "(" + str(self.X) + "," + str(self.Y) + ")"
        
        
            
# self.game should be automatically updated when changes outside of this scope,
# (passed as a deep copy)
class Agent:
    def __init__(self, game):
        self.game = game
        self.state = AgentState(game.getGridSize())
        self.batchSize = 1
        self.observationSize = len(self.state.asVector())
        self.actionSize = 10
        
        self.states = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(self.batchSize,), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(self.batchSize,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(self.batchSize,), name='done')

        self.network = DeepNetwork()
        
    def move(self):
        # use leared policy here to decide move..
        # move = Learning.move(self.state)
        self.state.updatePosition(2, 2)
        return Action(Action.PAWN, 2, 2)
    
    
        
    def updateFromEnemyMove(self, move, X, Y):
        if move.getType() == Action.WALL:
            self.state.addWall(X, Y)
        elif move.getType() == Action.PAWN:
            self.state.updateEnemyPosition(X, Y)
        else:
            raise ValueError("Invalid action")
            
        #print("move: ", move)
        #print("state: ", self.state)
        
        
        
        
class TopAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game)
        self.q = self.network.nn(self.states, [32, 32, self.actionSize], scope_name='Top_Q_primary')
        self.q_target = self.network.nn(self.states_next, [32, 32, self.actionSize], scope_name='Top_Q_target')
        
        # The prediction by the primary Q network for the actual actions.
        action_one_hot = tf.one_hot(self.actions, self.actionSize, 1.0, 0.0, name='top_action_one_hot')
        self.pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=-1, name='top_q_acted')
        
    def move(self, sess):
        move = Agent.move(self)
        # convert this "TopAgent perspective "move to "game perspective" move
        XRelativeToGame = self.game.getGridSize() - move.getX() - 1
        return Action(move.getType(), XRelativeToGame, move.getY())
        
        
        
        
    def updateFromEnemyMove(self, move):
        XRelativeToMe = self.game.getGridSize() - move.getX() - 1
        Agent.updateFromEnemyMove(self, move, XRelativeToMe, move.getY())
        

        
        
class BottomAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game)
        self.q = self.network.nn(self.states, [32, 32, self.actionSize], scope_name='Bot_Q_primary')
        self.q_target = self.network.nn(self.states_next, [32, 32, self.actionSize], scope_name='Bot_Q_target')
        
        # The prediction by the primary Q network for the actual actions.
        #action_one_hot = tf.one_hot(self.actions, self.actionSize, 1.0, 0.0, name='bot_action_one_hot')
        #self.pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=-1, name='bot_q_acted')
        
        
    def move(self, sess):
        move = Agent.move(self)
        # convert this "BottomAgent perspective "move to "game perspective" move
        #YRelativeToGame = self.game.getGridSize() - move.getY() - 1
        #return Action(move.getType(), move.getX(), YRelativeToGame)
        print(self.state)
        print(self.state.asVector())
        q = sess.run(self.q, feed_dict=
                     {self.states: np.asarray(self.state.asVector(), dtype=float).reshape(1,self.observationSize)})
        print(q)
        # The prediction by the primary Q network for the actual actions.
        action_one_hot = tf.one_hot(self.actions, self.actionSize, 1.0, 0.0, name='bot_action_one_hot')
        print(action_one_hot)
        self.pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=-1, name='bot_q_acted')
        print(self.pred)
        
        
        
        
    def updateFromEnemyMove(self, move):
        YRelativeToMe = self.game.getGridSize() - move.getY() - 1
        Agent.updateFromEnemyMove(self, move, move.getX(), YRelativeToMe)

        
        
        
        
        
        
        
        