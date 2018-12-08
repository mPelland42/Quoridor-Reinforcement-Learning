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
    WALL_HORIZONTAL = 1
    WALL_VERTICAL = 2
    AGENT_TOP = 3
    AGENT_BOT = 4
    OFF_GRID = 5


class GameState:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.intersections = [[0 for x in range(gridSize-1)] for x in range(gridSize-1)]
        topAgentPosition = Point(math.floor(gridSize/2), 0)
        botAgentPosition = Point(math.floor(gridSize/2), gridSize-1)

        #use this now to get agent positions.
        self.agentPositions = {BoardElement.AGENT_TOP: topAgentPosition, BoardElement.AGENT_BOT: botAgentPosition}

        self.walls = {BoardElement.AGENT_TOP: 10, BoardElement.AGENT_BOT: 10}
        self.movesTaken = 0
        self.wallPositions = []

        self.winner = None


    def updateAgentPosition(self, agentType, position):
        if agentType == BoardElement.AGENT_TOP:
            self.agentPositions[BoardElement.AGENT_TOP] = position
            if position.Y == self.gridSize - 1:
                self.winner = BoardElement.AGENT_TOP

        elif agentType == BoardElement.AGENT_BOT:
            self.agentPositions[BoardElement.AGENT_BOT] = position
            if position.Y == 0:
                self.winner = BoardElement.AGENT_BOT
        self.movesTaken += 1

    def getPosition(self, agentType):
        if agentType == BoardElement.AGENT_TOP:
            return copy.copy(self.agentPositions[BoardElement.AGENT_TOP])
        elif agentType == BoardElement.AGENT_BOT:
            return copy.copy(self.agentPositions[BoardElement.AGENT_BOT])

    def getMovesTaken(self):
        return self.movesTaken

    def addIntersection(self, position, orientation):
        self.wallPositions.append((position, orientation))
        self.intersections[position.X][position.Y] = orientation
        self.movesTaken += 1


    def removeWallCount(self, agentType):
        self.walls[agentType] -= 1

    def getWallCount(self, agentType):
        return self.walls[agentType]


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
            v.append(self.gridSize - self.agentPositions[BoardElement.AGENT_TOP].X - 1)
            v.append(self.agentPositions[BoardElement.AGENT_TOP].Y)

            v.append(self.gridSize - self.agentPositions[BoardElement.AGENT_BOT].X - 1)
            v.append(self.agentPositions[BoardElement.AGENT_BOT].Y)

            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_TOP])
            v.append(self.walls[BoardElement.AGENT_BOT])
            v.append(self.movesTaken)



        elif agentType == BoardElement.AGENT_BOT:
            for x in range(len(self.intersections)):
                for y in reversed(range(len(self.intersections[x]))):
                    v.append(self.intersections[x][y])

            # my position, then enemy position
            v.append(self.agentPositions[BoardElement.AGENT_BOT].X)
            v.append(self.gridSize - self.agentPositions[BoardElement.AGENT_BOT].Y - 1)

            v.append(self.agentPositions[BoardElement.AGENT_TOP].X)
            v.append(self.gridSize - self.agentPositions[BoardElement.AGENT_TOP].Y - 1)



            # my walls, then enemy walls
            v.append(self.walls[BoardElement.AGENT_BOT])
            v.append(self.walls[BoardElement.AGENT_TOP])
            v.append(self.movesTaken)


        return np.array(v)



    def __str__(self):
        s = "walls: \n"
        for y in range(self.gridSize-1):
            for x in range(self.gridSize-1):
                s += str(self.intersections[x][y])
            s+= "\n"

        s += "agentPositions[BoardElement.AGENT_TOP]: " + str(self.agentPositions[BoardElement.AGENT_TOP]) + "\n"
        s += "agentPositions[BoardElement.AGENT_BOT]: "+ str(self.agentPositions[BoardElement.AGENT_BOT]) + "\n"
        s += "Top Agent's walls left: " + str(self.walls[BoardElement.AGENT_TOP]) + "\n"
        s += "Bot agent's walls left: " + str(self.walls[BoardElement.AGENT_BOT]) + "\n"
        return s
