# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:54:50 2018

@author: Sam
"""

#from Game import Qoridor

from GameState import BoardElement
from Point import Point
import numpy as np

import tensorflow as tf
import copy
import random



moveProbability = 0.90



class Action:
    PAWN = "PAWN"
    WALL = "WALL"
    
    NUM_DIRECTIONS = 4
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    JUMP_UP = "UP"
    JUMP_DOWN = "DOWN"
    JUMP_LEFT = "LEFT"
    JUMP_RIGHT = "RIGHT"
    #STAY = "STAY"
    
    
    def __init__(self, actionType, direction = None, orientation = None, position = None):
            self.actionType = actionType
            self.direction = direction
            self.orientation = orientation
            self.position = position
            
            
    def makeAllActions(gridSize):
        
        # define all possible actions
        allActions = list()
        allActions.append(Action(Action.PAWN, Action.UP))
        allActions.append(Action(Action.PAWN, Action.DOWN))
        allActions.append(Action(Action.PAWN, Action.LEFT))
        allActions.append(Action(Action.PAWN, Action.RIGHT))
        allActions.append(Action(Action.PAWN, Action.JUMP_UP))
        allActions.append(Action(Action.PAWN, Action.JUMP_DOWN))
        allActions.append(Action(Action.PAWN, Action.JUMP_LEFT))
        allActions.append(Action(Action.PAWN, Action.JUMP_RIGHT))
        #allActions.append(Action(Action.PAWN, Action.STAY))
        
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
        
        elif self.direction == Action.JUMP_UP:
            self.position.addToY(2)
        elif self.direction == Action.JUMP_DOWN:
            self.position.addToY(-2)
        elif self.direction == Action.JUMP_RIGHT:
            self.position.addToX(2)
        elif self.direction == Action.JUMP_LEFT:
            self.position.addToX(-2)
    
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
    def __init__(self, game, agentType, sess, model, memory):
        self.game = game
        self.sess = sess
        self.model = model
        self.memory = memory
        self.agentType = agentType
        gridSize = game.getGridSize()
        
        self.allActions = Action.makeAllActions(gridSize)
        self.actionSize = len(self.allActions)
        



        
    def getType(self):
        return self.agentType
    
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



    
    def move(self, agentType, currentStateVector, epsilon):
        #print("Agent: ", agentType)
        
        
        # exclude illegal moves for now
        # might need to implement a negative reward for them if performance sucks
        
        
        if self.game.randomActions and (random.random() < epsilon):
            count = 0
            while True:
                if random.random() < moveProbability:
                    randomAction = random.randint(0, 3)
                else:
                    randomAction = random.randint(8, self.model.getNumActions() - 1)
                
                count += 1
                if count > 100:
                    # try the jumps
                    for i in range(4):
                        if not self.invalidMove(i + 4, agentType, self.game.getState()):
                            print("an AI jumped!")
                            return i + 4
                    
                    print("BUG")
                    # if not, must be some weird bug, return None to avoid a catastrophe
                    return -1, None
                    
                if not self.invalidMove(randomAction, agentType, self.game.getState()):
                    break
                
            return randomAction, self.allActions[randomAction]
        
        else:
            q = self.model.predictOne(currentStateVector, self.sess)
            q = q.flatten()
            values, indices = self.sess.run(tf.nn.top_k(q, len(q)))
            i = 0
            try:
                while self.invalidMove(indices[i], agentType, self.game.getState()):
                    #self.memory.addSample((self.game.getState().asVector(self.agentType),\
                                           #indices[i], -10, None))
                    i += 1
            except IndexError:
                #print("ERROR!")
                #print("q: ", q)
                #print("indices: ", indices)
                #print("indicesLEN: ", len(indices))
                #print("values: ", values)
                #print("all actions: ")
                #for a in self.allActions: print(a)
                #print("i:", i)
                #print("agentType: ", agentType)
                #print("state: ", self.game.getState())
                ##print("action: ", self.allActions[indices[i]])
                #self.game.drawError()
                return -1, None
            chosenAction = indices[i]
            return chosenAction, self.allActions[chosenAction]
            
    
    

    

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
                elif action.getDirection() == Action.JUMP_LEFT:
                    position.addToX(2)
                elif action.getDirection() == Action.JUMP_RIGHT:
                    position.addToX(-2)
                elif action.getDirection() == Action.JUMP_UP:
                    position.addToY(2)
                elif action.getDirection() == Action.JUMP_DOWN:
                    position.addToY(-2)
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
                elif action.getDirection() == Action.JUMP_LEFT:
                    position.addToX(-2)
                elif action.getDirection() == Action.JUMP_RIGHT:
                    position.addToX(2)
                elif action.getDirection() == Action.JUMP_UP:
                    position.addToY(-2)
                elif action.getDirection() == Action.JUMP_DOWN:
                    position.addToY(2)
            else:
                position = Point(position.X, self.game.getGridSize() - position.Y - 2)
                
                
        action.updatePosition(position)
        return action

    def learn(self):
        batch = self.memory.sample(self.model.getBatchSize())
        
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.model.getNumStates()) 
                                 if val[3] is None else val[3]) for val in batch])
    
        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predictBatch(states, self.sess)
        
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predictBatch(next_states, self.sess)
        
        # setup training arrays
        x = np.zeros((len(batch), self.model.getNumStates()))
        y = np.zeros((len(batch), self.model.getNumActions()))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + 0.9 * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self.model.trainBatch(self.sess, x, y)
        
        
class TopAgent(Agent):
    def __init__(self, game, sess, model, memory):
        Agent.__init__(self, game, BoardElement.AGENT_TOP, sess, model, memory)



    def move(self, epsilon):
        actionTuple = Agent.move(self, BoardElement.AGENT_TOP, self.game.getState().asVector(BoardElement.AGENT_TOP), epsilon)
        
        #print("game ", actionTuple[1])
        return actionTuple



class BottomAgent(Agent):
    def __init__(self, game, sess, model, memory):
        Agent.__init__(self, game, BoardElement.AGENT_BOT, sess, model, memory)

    def move(self, epsilon):
        actionTuple = Agent.move(self, BoardElement.AGENT_BOT, self.game.getState().asVector(BoardElement.AGENT_BOT), epsilon)
        
        #print("game ", actionTuple[1])
        return actionTuple
    
    
        
        
        