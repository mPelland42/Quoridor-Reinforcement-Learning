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
    WALL = 0
    EMPTY = 1
    AGENT_TOP = "T"
    AGENT_BOT = "B"
    WALL_HORIZONTAL = 2
    WALL_VERTICAL = 3
    OFF_GRID = 4

class GameState:
    def __init__(self, gridSize, numWalls, maxActions):
        self.gridSize = gridSize
        self.maxActions = maxActions
        self.fullGridSize = 2 * gridSize - 1
        self.grid = [[BoardElement.EMPTY for x in range(self.fullGridSize)] for x in range(self.fullGridSize)]
        
        self.intersections = [[BoardElement.EMPTY for x in range(gridSize-1)] for x in range(gridSize-1)]
        
        topAgentPosition = Point(math.floor(gridSize/2), 0)
        botAgentPosition = Point(math.floor(gridSize/2), gridSize-1)
        
        self.grid[topAgentPosition.X * 2][topAgentPosition.Y * 2] = BoardElement.AGENT_TOP
        self.grid[botAgentPosition.X * 2][botAgentPosition.Y * 2] = BoardElement.AGENT_BOT

        #use this now to get agent positions.
        self.agentPositions = {BoardElement.AGENT_TOP: topAgentPosition, BoardElement.AGENT_BOT: botAgentPosition}

        self.walls = {BoardElement.AGENT_TOP: numWalls, BoardElement.AGENT_BOT: numWalls}
        self.movesTaken = 0
        self.wallPositions = []
        self.previousActions = {BoardElement.AGENT_TOP: list(), BoardElement.AGENT_BOT: list()}
        for i in range(self.maxActions):
            self.previousActions[BoardElement.AGENT_TOP].append(-1)
            self.previousActions[BoardElement.AGENT_BOT].append(-1)
            

        self.memoryOfStates = {BoardElement.AGENT_TOP: list(), BoardElement.AGENT_BOT: list()}
        self.winner = None


    def updateAgentPosition(self, agentType, position):
        if agentType == BoardElement.AGENT_TOP:
            self.grid[self.agentPositions[BoardElement.AGENT_TOP].X * 2]\
            [self.agentPositions[BoardElement.AGENT_TOP].Y * 2] = BoardElement.EMPTY
            
            self.grid[position.X * 2][position.Y * 2] = BoardElement.AGENT_TOP
            
            self.agentPositions[BoardElement.AGENT_TOP] = position
            if position.Y == self.gridSize - 1:
                self.winner = BoardElement.AGENT_TOP
                self.fillInMemory(BoardElement.AGENT_TOP, BoardElement.AGENT_BOT)


        elif agentType == BoardElement.AGENT_BOT:
            self.grid[self.agentPositions[BoardElement.AGENT_BOT].X * 2]\
            [self.agentPositions[BoardElement.AGENT_BOT].Y * 2] = BoardElement.EMPTY
            
            self.grid[position.X * 2][position.Y * 2] = BoardElement.AGENT_BOT
            
            self.agentPositions[BoardElement.AGENT_BOT] = position
            if position.Y == 0:
                self.winner = BoardElement.AGENT_BOT
                self.fillInMemory(BoardElement.AGENT_BOT, BoardElement.AGENT_TOP)
                
            
        self.movesTaken += 1
        

    def getPosition(self, agentType):
        if agentType == BoardElement.AGENT_TOP:
            return copy.copy(self.agentPositions[BoardElement.AGENT_TOP])
        elif agentType == BoardElement.AGENT_BOT:
            return copy.copy(self.agentPositions[BoardElement.AGENT_BOT])

    def addMove(self):
        self.movesTaken += 1
    
    def getMovesTaken(self):
        return self.movesTaken

    def getIntersection(self, position):
        return self.intersections[position.X][position.Y]
    
    def addIntersection(self, position, orientation):
        
        self.wallPositions.append((position, orientation))
        
        if orientation == BoardElement.WALL_HORIZONTAL:
            self.grid[position.X*2][position.Y*2 + 1] = BoardElement.WALL
            self.grid[position.X*2 + 1][position.Y*2 + 1] = BoardElement.WALL
            self.grid[position.X*2 + 2][position.Y*2 + 1] = BoardElement.WALL
            
            if position.X != 0:
                # fill in the cracks
                # left
                if self.grid[position.X*2 - 2][position.Y*2 + 1] == BoardElement.WALL:
                    self.grid[position.X*2 - 1][position.Y*2 + 1] = BoardElement.WALL
                    
            if position.X != self.gridSize-2:
                # right
                if self.grid[position.X*2 + 4][position.Y*2 + 1] == BoardElement.WALL:
                    self.grid[position.X*2 + 3][position.Y*2 + 1] = BoardElement.WALL
            
                    
        elif orientation == BoardElement.WALL_VERTICAL:
            self.grid[position.X*2 + 1][position.Y*2] = BoardElement.WALL
            self.grid[position.X*2 + 1][position.Y*2 + 1] = BoardElement.WALL
            self.grid[position.X*2 + 1][position.Y*2 + 2] = BoardElement.WALL
            
            if position.Y != 0:
                # down
                if self.grid[position.X*2 + 1][position.Y*2 - 2] == BoardElement.WALL:
                    self.grid[position.X*2 + 1][position.Y*2 - 1] = BoardElement.WALL
            
            if position.Y != self.gridSize-2:
                # up
                if self.grid[position.X*2 + 1][position.Y*2 + 4] == BoardElement.WALL:
                    self.grid[position.X*2 + 1][position.Y*2 + 3] = BoardElement.WALL
        
            
        self.intersections[position.X][position.Y] = orientation
        self.movesTaken += 1
        
    def removeWallCount(self, agentType):
        self.walls[agentType] -= 1

    def getWallCount(self, agentType):
        return self.walls[agentType]


    def getWinner(self):
        return self.winner

    def addState(self, stateVector, agentType):
        self.memoryOfStates[agentType].append((stateVector, None))
        
    def fillInMemory(self, winner, loser):
        for i in range(len(self.memoryOfStates[winner])):
            self.memoryOfStates[winner][i] = (self.memoryOfStates[winner][i][0], 1)
            
        for i in range(len(self.memoryOfStates[loser])):
            self.memoryOfStates[loser][i] = (self.memoryOfStates[loser][i][0], 0)

    def addAction(self, actionIndex, agentType):
        self.previousActions[agentType].append(actionIndex)
        del self.previousActions[agentType][0]
        
    def getStateMemory(self):
        return self.memoryOfStates

    # vector inputs into the neural net need to look identical
    # so top and bot have the same seperate but fair perspectives of the game
    def asVector(self, agentType):
        v = []
        if agentType == BoardElement.AGENT_TOP:
            #for y in range(self.gridSize-1):
            for y in range(self.fullGridSize):
                for x in reversed(range(self.fullGridSize)):
                    #v.append(self.intersections[x][y])
                    
                    if self.grid[x][y] == BoardElement.AGENT_TOP:
                        v.append(2)
                    elif self.grid[x][y] == BoardElement.AGENT_BOT:
                        v.append(3)
                    else:
                        v.append(self.grid[x][y])
                    
                    
            # my position then enemy position
            #v.append((self.gridSize - 1) - self.getPosition(BoardElement.AGENT_TOP).X)
            #v.append(self.getPosition(BoardElement.AGENT_TOP).Y)
            
            #v.append((self.gridSize - 1) - self.getPosition(BoardElement.AGENT_BOT).X)
            #v.append(self.getPosition(BoardElement.AGENT_BOT).Y)
            
            
            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_TOP])
            v.append(self.walls[BoardElement.AGENT_BOT])
            
            # recent actions
            #for a in self.previousActions[BoardElement.AGENT_TOP]:
            #    v.append(a)
            
            #v.append(self.movesTaken)



        elif agentType == BoardElement.AGENT_BOT:
            #for y in reversed(range(self.gridSize-1)):
            for y in reversed(range(self.fullGridSize)):
                for x in range(self.fullGridSize):
                    #v.append(self.intersections[x][y])
                    
                    if self.grid[x][y] == BoardElement.AGENT_BOT:
                        v.append(2)
                    elif self.grid[x][y] == BoardElement.AGENT_TOP:
                        v.append(3)
                    else:
                        v.append(self.grid[x][y])
                    

            #v.append(self.getPosition(BoardElement.AGENT_BOT).X)
            #v.append((self.gridSize - 1) - self.getPosition(BoardElement.AGENT_BOT).Y)
            
            #v.append(self.getPosition(BoardElement.AGENT_TOP).X)
            #v.append((self.gridSize - 1) - self.getPosition(BoardElement.AGENT_TOP).Y)
            

            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_BOT])
            v.append(self.walls[BoardElement.AGENT_TOP])
            
            #v.append(self.movesTaken)
            
            
            # recent actions
            #for a in self.previousActions[BoardElement.AGENT_BOT]:
            #    v.append(a)

        v = np.array(v)
        self.addState(v, agentType)
        return v



    def __str__(self):
        s = "grid: \n"
        for y in range(self.fullGridSize):
            for x in range(self.fullGridSize):
                s += str(self.grid[x][y])
            s+= "\n"
        s += "agentPositions[BoardElement.AGENT_TOP]: " + str(self.agentPositions[BoardElement.AGENT_TOP]) + "\n"
        s += "agentPositions[BoardElement.AGENT_BOT]: "+ str(self.agentPositions[BoardElement.AGENT_BOT]) + "\n"
        s += "Top Agent's walls left: " + str(self.walls[BoardElement.AGENT_TOP]) + "\n"
        s += "Bot agent's walls left: " + str(self.walls[BoardElement.AGENT_BOT]) + "\n"
        return s
