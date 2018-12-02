# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:55:27 2018

@author: Sam

AgentState is the state of the game from this agent's perspective

"""

import math

class AgentState:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.intersections = [[0 for x in range(gridSize-1)] for x in range(gridSize-1)]
        self.position = (math.floor(gridSize/2), 0)
        self.enemyPosition = (math.floor(gridSize/2), gridSize-1)
        self.wallsLeft = 10
        self.enemyWallsLeft = 10
        
    def addMyWall(self, X, Y):
        self.intersections[X][Y] = 1
        self.wallsLeft -= 1
        
    def addEnemyWall(self, X, Y):
        self.intersections[X][Y] = 1
        self.enemyWallsLeft -= 1
        
    def updatePosition(self, X, Y):
        self.position = (X, Y)
        
    def updateEnemyPosition(self, X, Y):
        self.enemyPosition = (X, Y)
        
        
    def asVector(self):
        v = []
        for x in range(len(self.intersections)):
            for y in range(len(self.intersections[x])):
                v.append(self.intersections[x][y])
        v.append(self.position[0])
        v.append(self.position[1])
        v.append(self.enemyPosition[0])
        v.append(self.enemyPosition[1])
        v.append(self.wallsLeft)
        v.append(self.enemyWallsLeft)
        return v
        
        
        
    def __str__(self):
        s = "intersections: " + " ".join(str(x) for x in self.intersections) +"\n"
        s += "position: " + " ".join(str(x) for x in self.position)+ "\n"
        s += "enemy pos: "+ " ".join(str(x) for x in self.enemyPosition) + "\n"
        s += "wallsLeft: " + str(self.wallsLeft) + "\n"
        s += "enemyWallsLeft: " + str(self.enemyWallsLeft) + "\n"
        return s