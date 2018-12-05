# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:54:50 2018

@author: Sam
"""

#from Game import Qoridor

from GameState import GameState
from GameState import BoardElement
from DeepNetwork import DeepNetwork
from Point import Point
import numpy as np

import tensorflow as tf
import copy




class Action:
    PAWN = "PAWN"
    WALL = "WALL"
    
    NUM_DIRECTIONS = 5
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"
    
    
    
    def __init__(self, actionType, direction = None, orientation = None, position = None):
            self.actionType = actionType
            self.direction = direction
            self.orientation = orientation
            self.position = position
            
            
    def makeAllActions(gridSize):
        
        # define all possible actions
        allActions = []
        allActions.append(Action(Action.PAWN, Action.UP))
        allActions.append(Action(Action.PAWN, Action.DOWN))
        allActions.append(Action(Action.PAWN, Action.LEFT))
        allActions.append(Action(Action.PAWN, Action.RIGHT))
        allActions.append(Action(Action.PAWN, Action.STAY))
        
        for x in range(gridSize - 1):
            for y in range(gridSize - 1):
                allActions.append(Action(Action.WALL, None, BoardElement.WALL_HORIZONTAL, Point(x, y)))
                allActions.append(Action(Action.WALL, None, BoardElement.WALL_VERTICAL, Point(x, y)))
        return allActions
        
        
        
    def getType(self):
        return copy.copy(self.actionType)
    
    def getDirection(self):
        return copy.copy(self.direction)
    
    def getOrientation(self):
        return copy.copy(self.orientation)
    
    def getPosition(self):
        return copy.copy(self.position)
    
    def updatePosition(self, position):
        self.position = position
        
    def applyDirection(self):
        
        if self.direction == Action.UP:
            self.position.addToY(1)
        elif self.direction == Action.DOWN:
            self.position.addToY(-1)
        elif self.direction == Action.RIGHT:
            self.position.addToX(1)
        elif self.direction == Action.LEFT:
            self.position.addToX(-1)
    
    def xstr(self,s):
        if s is None:
            return 'NULL'
        return str(s)

    def __str__(self):
        return "Action: " + self.xstr(self.actionType) + " Direction: " + self.xstr(self.direction) + \
    " Orientation: " + self.xstr(self.orientation) + "  X,Y:" + str(self.position)
    
        
        
            
# self.game should be automatically updated when changes outside of this scope,
# (passed as a deep copy)
class Agent:
    def __init__(self, game, agentType):
        self.game = game
        gridSize = game.getGridSize()
        
        self.batchSize = 1                 # both bot and top have the same vector size
        self.observationSize = len(game.getState().asVector(BoardElement.AGENT_TOP))
        
        
        self.allActions = Action.makeAllActions(gridSize)
        self.actionSize = len(self.allActions)
        
        
        
        
        self.states = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(self.batchSize,self.actionSize), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(self.batchSize,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(self.batchSize,), name='done')

        self.network = DeepNetwork()
        
        if agentType == BoardElement.AGENT_TOP:
            networkName = "TOP_Q_primary"
        elif agentType == BoardElement.AGENT_BOT:
            networkName = "BOT_Q_primary"
            
        self.q = self.network.nn(self.states, [32, 32, self.actionSize], scope_name = networkName)
        self.softmax = tf.nn.softmax(self.q)
        #print(self.softmax)
        #self.q_target = self.network.nn(self.states_next, [32, 32, self.actionSize], scope_name='Bot_Q_target')
        
        
        
    def invalidMove(self, index, agentType, gameState):
        action = self.allActions[index]
        action = self.makeActionReadyForGame(agentType, action, gameState)
        
        orientation = 0
        
        if action.getType() == Action.PAWN:
            moveType = 'p'
            
        elif action.getType() == Action.WALL:
            moveType = 'w'
            if action.getOrientation() == BoardElement.WALL_VERTICAL:
                orientation = 1
            elif action.getOrientation() == BoardElement.WALL_HORIZONTAL:
                orientation = 2
                
        if agentType == BoardElement.AGENT_TOP:
            agentNumber = 0
        elif agentType == BoardElement.AGENT_BOT:
            agentNumber = 1
            
        move = (moveType, (action.getPosition().X, action.getPosition().Y), orientation)
        return not self.game.isLegalMove(agentNumber, move)



    
    def move(self, agentType, currentStateVector, sess):
        print("Agent: ", agentType)
        
        
        
        # use leared policy here to decide move..
        q = sess.run(self.softmax, feed_dict=
                     {self.states: np.asarray(currentStateVector, dtype=float).reshape(1,self.observationSize)})
        q = q.flatten()
        """ q:         0.23           0.1            0.6            0.07     """
        
        
        values, indices = sess.run(tf.nn.top_k(q, len(q)-1))
        
        
        # now, filter out invalid moves
        i = 0
        while self.invalidMove(indices[i], agentType, self.game.getState()):
            q[indices[i]] = 0
            i += 1
            
        
        actionIndex = indices[i]
        action = self.allActions[actionIndex]
        
        return action
        
    

    def makeActionReadyForGame(self, agentType, action, gameState):
        
        position = action.getPosition()
        
        if agentType == BoardElement.AGENT_TOP:
            if action.getType() == Action.PAWN:
                position = gameState.getPosition(BoardElement.AGENT_TOP)
                
                # reversed horizontally, since this is the top agent's perspective
                if action.getDirection() == Action.LEFT:
                    position.addToX(1)
                elif action.getDirection() == Action.RIGHT:
                    position.addToX(-1)
                elif action.getDirection() == Action.UP:
                    position.addToY(1)
                elif action.getDirection() == Action.DOWN:
                    position.addToY(-1)
            else:
                position = Point(self.game.getGridSize() - position.X - 2, position.Y)

            
        elif agentType == BoardElement.AGENT_BOT:
            if action.getType() == Action.PAWN:
                position = gameState.getPosition(BoardElement.AGENT_BOT)
                
                # reversed horizontally, since this is the top agent's perspective
                if action.getDirection() == Action.LEFT:
                    position.addToX(-1)
                elif action.getDirection() == Action.RIGHT:
                    position.addToX(1)
                elif action.getDirection() == Action.UP:
                    position.addToY(-1)
                elif action.getDirection() == Action.DOWN:
                    position.addToY(1)
            else:
                position = Point(position.X, self.game.getGridSize() - position.Y - 2)
                
                
        action.updatePosition(position)
        return action


        
        
class TopAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game, BoardElement.AGENT_TOP)



    def move(self, sess):
        action = Agent.move(self, BoardElement.AGENT_TOP, self.game.getState().asVector(BoardElement.AGENT_TOP), sess)
        
        print("game ", action)
        return action



class BottomAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game, BoardElement.AGENT_BOT)

    def move(self, sess):
        action = Agent.move(self, BoardElement.AGENT_BOT, self.game.getState().asVector(BoardElement.AGENT_BOT), sess)
        
        print("game ", action)
        return action
    
    
        
        
        