# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:55:27 2018

@author: Sam

AgentState is the state of the game from this agent's perspective

"""

import math
from Point import Point
import copy
import numpy as np

class BoardElement():
    EMPTY = 0
    AGENT_TOP = 1
    AGENT_BOT = 2
    WALL_HORIZONTAL = 3
    WALL_VERTICAL = 4
    

class GameState:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.intersections = [[0 for x in range(gridSize-1)] for x in range(gridSize-1)]
        self.topAgentPosition = Point(math.floor(gridSize/2), 0)
        self.botAgentPosition = Point(math.floor(gridSize/2), gridSize-1)
        
        self.walls = {BoardElement.AGENT_TOP: 10, BoardElement.AGENT_BOT: 10}
        
        self.winner = None
        
        
    def updateAgentPosition(self, agentType, position):
        if agentType == BoardElement.AGENT_TOP:
            self.topAgentPosition = position
            if position.Y == self.gridSize - 1:
                self.winner = BoardElement.AGENT_TOP
                
        elif agentType == BoardElement.AGENT_BOT:
            self.botAgentPosition = position
            if position.Y == 0:
                self.winner = BoardElement.AGENT_BOT
                
    def getPosition(self, agentType):
        if agentType == BoardElement.AGENT_TOP:
            return copy.copy(self.topAgentPosition)
        elif agentType == BoardElement.AGENT_BOT:
            return copy.copy(self.botAgentPosition)
    
    def addIntersection(self, position, orientation):
        self.intersections[position.X][position.Y] = orientation
    
    def removeWallCount(self, agentType):
        self.walls[agentType] -= 1
    
    
    def getWinner(self):
        return self.winner
        
        
    # vector inputs into the neural net need to look identical
    # so top and bot have the same seperate but fair perspectives of the game
    def asVector(self, agentType):
        v = []
        if agentType == BoardElement.AGENT_TOP:
            for x in reversed(range(len(self.intersections))):
                for y in range(len(self.intersections[x])):
                    v.append(self.intersections[x][y])
                    
            # my position, then enemy position
            v.append(self.gridSize - self.topAgentPosition.X - 1)
            v.append(self.topAgentPosition.Y)
                     
            v.append(self.gridSize - self.botAgentPosition.X - 1)
            v.append(self.botAgentPosition.Y)
            
            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_TOP])
            v.append(self.walls[BoardElement.AGENT_BOT])
                        
                    
            
        elif agentType == BoardElement.AGENT_BOT:
            for x in range(len(self.intersections)):
                for y in reversed(range(len(self.intersections[x]))):
                    v.append(self.intersections[x][y])
                    
            # my position, then enemy position
            v.append(self.botAgentPosition.X)
            v.append(self.gridSize - self.botAgentPosition.Y - 1)
            
            v.append(self.topAgentPosition.X)
            v.append(self.gridSize - self.topAgentPosition.Y - 1)
                     

            
            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_BOT])
            v.append(self.walls[BoardElement.AGENT_TOP])
            
            
        return np.array(v)
        
        
        
    def __str__(self):
        s = "intersections: " + " ".join(str(x) for x in self.intersections) +"\n"
        s += "TopAgentPosition: " + str(self.topAgentPosition) + "\n"
        s += "BotAgentPosition: "+ str(self.botAgentPosition) + "\n"
        s += "Top Agent's walls left: " + str(self.walls[BoardElement.AGENT_TOP]) + "\n"
        s += "Bot agent's walls left: " + str(self.walls[BoardElement.AGENT_BOT]) + "\n"
        return s
    
    
    