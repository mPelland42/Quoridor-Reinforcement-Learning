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
    
    NUM_DIRECTIONS = 5
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"
    
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    
    
    def __init__(self, actionType, direction = None, orientation = None, X = None, Y = None, newX = None, newY = None):
            self.actionType = actionType
            self.direction = direction
            self.orientation = orientation
            self.X = X
            self.Y = Y
            self.newX = newX
            self.newY = newY
            
    def getType(self):
        return self.actionType
    
    def getDirection(self):
        return self.direction
    
    def getOrientation(self):
        return self.orientation
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    
    def getNewX(self):
        return self.newX
    
    def getNewY(self):
        return self.newY
    
    def applyDirection(self):
        self.newX = self.X
        self.newY = self.Y
        
        if self.direction == Action.UP:
            self.newY += 1
        elif self.direction == Action.DOWN:
            self.newY -= 1
        elif self.direction == Action.RIGHT:
            self.newX += 1
        elif self.direction == Action.LEFT:
            self.newX -= 1
                
    def __str__(self):
        return "Action: " + self.actionType + " direction: " + self.direction + \
    "  X,Y:(" + str(self.X) + "," + str(self.Y) + ")" + "  New X,Y:(" + str(self.newX) + \
    "," + str(self.newY) + ")"
        
        
            
# self.game should be automatically updated when changes outside of this scope,
# (passed as a deep copy)
class Agent:
    def __init__(self, game, networkName):
        self.game = game
        gridSize = game.getGridSize()
        self.state = AgentState(gridSize)
        self.batchSize = 1
        self.observationSize = len(self.state.asVector())
        
        
        # define all possible actions
        self.allActions = []
        self.allActions.append(Action(Action.PAWN, Action.UP))
        self.allActions.append(Action(Action.PAWN, Action.DOWN))
        self.allActions.append(Action(Action.PAWN, Action.LEFT))
        self.allActions.append(Action(Action.PAWN, Action.RIGHT))
        self.allActions.append(Action(Action.PAWN, Action.STAY))
        
        for x in range(gridSize - 1):
            for y in range(gridSize - 1):
                self.allActions.append(Action(Action.WALL, None, X = x, Y = y))

        
        self.actionSize = len(self.allActions)
        
        self.states = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(self.batchSize, self.observationSize), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(self.batchSize,self.actionSize), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(self.batchSize,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(self.batchSize,), name='done')

        self.network = DeepNetwork()
        
        realNetworkName = networkName + "_Q_primary"
        self.q = self.network.nn(self.states, [32, 32, self.actionSize], scope_name=realNetworkName)
        self.softmax = tf.nn.softmax(self.q)
        print(self.softmax)
        #self.q_target = self.network.nn(self.states_next, [32, 32, self.actionSize], scope_name='Bot_Q_target')
        
        
        
    def invalidMove(self, index, state, agentType):
        action = self.allActions[index]
        print(action)

        orientation = 0
        
        if action.getType() == Action.PAWN:
            X, Y = self.state.getPosition()
            moveType = 'p'
            if action.getDirection() == Action.UP:
                Y += 1
            elif action.getDirection() == Action.DOWN:
                Y -= 1
            elif action.getDirection() == Action.RIGHT:
                X += 1
            elif action.getDirection() == Action.LEFT:
                X -= 1
        else:
            X = action.getX()
            Y = action.getY()
            moveType = 'w'
            if action.getOrientation() == Action.HORIZONTAL:
                orientation = 2
            else:
                orientation = 1
        
        
        if agentType == "bot":
            agentNumber = 1
            YRelativeToGame = self.game.getGridSize() - Y - 1
            XRelativeToGame = X
        else:
            agentNumber = 0
            XRelativeToGame = self.game.getGridSize() - X - 1
            YRelativeToGame = Y
            
            
        move = (moveType, (XRelativeToGame, YRelativeToGame), orientation)
        if self.game.isLegalMove(agentNumber, move) == True:
            print("INVALID\n")
        else:
            print("OK\n")
            
        return self.game.isLegalMove(agentNumber, move)






    
    def move(self, agentType, sess):
        print("Agent: ", agentType)
        


        # use leared policy here to decide move..
        q = sess.run(self.softmax, feed_dict=
                     {self.states: np.asarray(self.state.asVector(), dtype=float).reshape(1,self.observationSize)})
        q = q.flatten()
        """ q:         0.23           0.1            0.6            0.07     """
        
        
        # now, filter out invalid moves
        #print("q: ", q)
        print("Agent is at: ", self.state.getPosition())
        for i in range(len(q)):
            if self.invalidMove(i, self.state, agentType):
                q[i] = 0
        
        
        
        values, indices = sess.run(tf.nn.top_k(q, 1))
        #print("values: ", values)
        #print("indices: ", indices)
        actionIndex = indices[0]
        action = self.allActions[actionIndex]
        
        if(action.getType() == Action.PAWN):
            action.applyDirection()
            self.state.updatePosition(action.getNewX(), action.getNewY())
        
        return action
        
    
    
        
    def updateFromEnemyMove(self, move, X, Y):
        if move.getType() == Action.WALL:
            self.state.addWall(X, Y)
        elif move.getType() == Action.PAWN:
            self.state.updateEnemyPosition(X, Y)
        else:
            raise ValueError("Invalid action")
            

        
        
        
class TopAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game, "top")



    def move(self, sess):
        action = Agent.move(self, "top", sess)
        
        # convert this "BottomAgent perspective "move to "game perspective" move
        XRelativeToGame = self.game.getGridSize() - action.getX() - 1
        gameAction = Action(action.getType(), action.getDirection(), action.getOrientation(), XRelativeToGame, action.getY())
        return gameAction
        
        
    def updateFromEnemyMove(self, move):
        XRelativeToMe = self.game.getGridSize() - move.getX() - 1
        Agent.updateFromEnemyMove(self, move, XRelativeToMe, move.getY())
        
        
        

        
        
class BottomAgent(Agent):
    def __init__(self, game):
        Agent.__init__(self, game, "bot")

    def move(self, sess):
        action = Agent.move(self, "bot", sess)
        
        # convert this "BottomAgent perspective "move to "game perspective" move
        YRelativeToGame = self.game.getGridSize() - action.getY() - 1
        gameAction = Action(action.getType(), action.getDirection(), action.getOrientation(), action.getX(), YRelativeToGame)
        return gameAction
    
    
    
    
        
    def updateFromEnemyMove(self, move):
        YRelativeToMe = self.game.getGridSize() - move.getY() - 1
        Agent.updateFromEnemyMove(self, move, move.getX(), YRelativeToMe)

        
        
        
        
        
        