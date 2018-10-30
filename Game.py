import pygame
from AStar import AStar

class Qoridor:
    def __init__(self, screen):
        self.spaces = [[-1 for x in range(9)] for x in range(9)]
        self.intersections = [[0 for x in range(8)] for x in range(8)]
        self.agents = [(4, 0), (4, 8)]
        for i in range(len(self.agents)):
            self.spaces[self.agents[i][0]][self.agents[i][1]] = i
        self.agentColors = [(0, 0, 255), (0, 255, 0)]
        self.walls= []
        self.wallCounts = [10, 10]
        self.turn = 1
        self.screen = screen

    def getAllNeighbors(self, space):
        neighbors = []
        if space[0] > 0:
            neighbors.append((space[0] - 1, space[1]))
        if space[0] < 8:
            neighbors.append((space[0] + 1, space[1]))
        if space[1] > 0:
            neighbors.append((space[0], space[1] - 1))
        if space[1] < 8:
            neighbors.append((space[0], space[1] + 1))
        return neighbors

    def draw(self):
        boxSize = self.screen.get_width()/9
        shift = (self.screen.get_width() - boxSize*9)/2
        for i in range(9):
            for j in range(9):
                pygame.draw.rect(self.screen, (255, 0, 0), [i*boxSize + shift, j*boxSize + shift, boxSize, boxSize], 10)
        for i in range(len(self.agents)):
            pygame.draw.circle(self.screen, self.agentColors[i], (self.agents[i][0] * boxSize + boxSize/2 + shift, self.agents[i][1] * boxSize + boxSize/2 + shift), int(boxSize * .25))

    def getSpace(self, space):
        if space[0] > 0 and space[0] < 8 and space[1] > 0 and space[1] < 8:
            return self.spaces[space[0]][space[1]]
        else:
            return 7;

    def getPawnMoves(self, space):
        neighbors = []
        for neighbor in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            print [sum(x) for x in zip(space, neighbor)]
            if self.getSpace([sum(x) for x in zip(space, neighbor)]) == -1:
                neighbors.append(neighbor)
            elif self.getSpace([sum(x) for x in zip(space, (2*x for x in neighbor))]) == -1:
                neighbors.append((2*x for x in neighbor))
            else: #now check diagonal jumps.
                if neighbor[0] == 0:
                    if self.getSpace([sum(x) for x in zip(space, (1, neighbor[1]))]) == -1:
                        neighbors.append((1, neighbor[1]))
                    if self.getSpace([sum(x) for x in zip(space, (-1, neighbor[1]))]) == -1:
                        neighbors.append((-1, neighbor[1]))
                else:
                    if self.getSpace([sum(x) for x in zip(space, (neighbor[0], 1))]) == -1:
                        neighbors.append((neighbor[0], 1))
                    if self.getSpace([sum(x) for x in zip(space, (neighbor[0], -1))]) == -1:
                        neighbors.append((neighbor[0], -1))
        return neighbors

    def canMoveTo(self, start, end):
        if start[0] == end[0]:
            if(start[0] - 1 >= 0) and self.intersections[start[0] - 1][min(start[1], end[1])] == 2:
                return False
            if(start[0] <= 8 and self.intersections[start[0]][min(start[1], end[1])] == 2):
                return False
        else:
            if(start[1] - 1 >= 0) and self.intersections[min(start[0], end[0])][start[1] - 1] == 1:
                return False
            if(start[1] <= 8 and self.intersections[min(start[0], end[0])][start[1]] == 1):
                return False
        return True

    def move(self, agent, target):
        oldPos = self.agents[agent]
        self.spaces[oldPos[0]][oldPos[1]] = -1
        self.spaces[target[0]][target[1]] = agent
        self.agents[agent] = target

    def getLegalMoves(self, agent):
        #first get legal pawn moves, designated by 'p' as first element.
        moves = []
        for item in self.getPawnMoves(self.agents[agent]):
            moves.append(('p', item[0], item[1]))
        for i in range(len(self.intersections)):
            for j in range(len(self.intersections[i])):
                if not self.isWall(i, j):
                    if not self.isWall(i, j + 1) == 1 and not self.isWall(i, j - 1) == 1:
                        moves.append(('w', i, j, 1))
                    if not self.isWall(i + 1, j) == 2 and not self.isWall(i - 1, j) == 2:
                            moves.append(('w', i, j, 2))
        return moves

    def isLegalMove(self, agent, move):
        if move[0] == 'p':
            return self.canMoveTo(self.agents[agent], (move[0], move[1]))
        else:
            if not self.isWall(move[1], move[2]):
                if move[3] == 1 and not self.isWall(move[0], move[0] + 1) == 1 and not self.isWall(move[0], move[1] - 1) == 1:
                    return True
                if move[3] == 2 and not self.isWall(move[0] + 1, move[1]) == 2 and not self.isWall(move[0] - 1, move[1]) == 2:
                    return True
        return False

    #edge is 0 or 8 to indicate which edge
    def pathExists(self, space, edge):
        AStar(self, space, lambda space : space[1] == edge, lambda space : abs(space[1] - edge))

    def isWall(self, x, y):
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return 0
        else:
            return self.intersections[x][y]