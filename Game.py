import pygame
from AStar import AStar

from Agents import Action

class Qoridor:
    def __init__(self, screen):
        self.spaces = [[-1 for x in range(9)] for x in range(9)]
        self.intersections = [[0 for x in range(8)] for x in range(8)]
        self.agents = [(4, 0), (4, 8)]
        for i in range(len(self.agents)):
            self.spaces[self.agents[i][0]][self.agents[i][1]] = i
        self.agentColors = [(0, 0, 255), (0, 255, 0)]
        self.walls = []
        for i in self.walls: #for debugging
            self.intersections[i[0]][i[1]] = i[2]
        self.wallCounts = [10, 10]
        self.turn = 1
        self.screen = screen
        
        
    def getGridSize(self):
        return 9
        

    #returns all neighboring cells
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

    #draws it to the screen
    def draw(self):
        boxSize = self.screen.get_width()/9
        shift = (self.screen.get_width() - boxSize*9)/2
        for i in range(9):
            for j in range(9):
                pygame.draw.rect(self.screen, (255, 0, 0), [i*boxSize + shift, j*boxSize + shift, boxSize, boxSize], 10)
        for i in range(len(self.agents)):
            pygame.draw.circle(self.screen, self.agentColors[i], (int(self.agents[i][0] * boxSize + boxSize/2 + shift), int(self.agents[i][1] * boxSize + boxSize/2 + shift)), int(boxSize * .25))
        for i in self.walls:
            if i[1] == 2:
                pygame.draw.rect(self.screen, (178, 178, 1), [i[0][0]*boxSize + shift + 5, (i[0][1] + 1)*boxSize + shift, boxSize*2 - 10, 1], 10)
            else:
                pygame.draw.rect(self.screen, (178, 178, 1), [(i[0][0] + 1) * boxSize + shift, i[0][1] * boxSize + shift + 5, 1, boxSize*2 - 10], 10)

    #gets value of a space, returns 7 if out of bounds
    def getSpace(self, space):
        if space[0] >= 0 and space[0] <= 8 and space[1] >= 0 and space[1] <= 8:
            return self.spaces[space[0]][space[1]]
        else:
            return 7;

    #adds 2d tuples
    def tupAdd(self, tup1, tup2):
        return (tup1[0] + tup2[0], tup1[1] + tup2[1])

    #gets all possible pawn moves
    def getPawnMoves(self, space):
        #print "getting pawn moves ", space
        neighbors = []
        for neighbor in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target = self.tupAdd(space, neighbor)
            if self.getSpace(target) == -1:
                if self.canMoveTo(space, target):
                    neighbors.append(neighbor)
            #if no direct move available, see if we can jump the pawn directly
            elif self.getSpace(self.tupAdd(space, [2*x for x in neighbor])) == -1:
                if self.canMoveTo(target, self.tupAdd(space, [2*x for x in neighbor])):
                    neighbors.append(tuple(2*x for x in neighbor))
            else: #now check diagonal jumps.
                if neighbor[0] == 0:
                    if self.getSpace(self.tupAdd(space, (1, neighbor[1]))) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(self.tupAdd(space, neighbor), self.tupAdd(space, (1, neighbor[1]))):
                            neighbors.append((1, neighbor[1]))
                    if self.getSpace([sum(x) for x in zip(space, (-1, neighbor[1]))]) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(self.tupAdd(space, neighbor), self.tupAdd(space, (-1, neighbor[1]))):
                            neighbors.append((1, neighbor[1]))
                else:
                    if self.getSpace([sum(x) for x in zip(space, (neighbor[0], 1))]) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(
                                self.tupAdd(space, neighbor), self.tupAdd(space, (neighbor[0], 1))):
                            neighbors.append((neighbor[0], 1))
                    if self.getSpace([sum(x) for x in zip(space, (neighbor[0], -1))]) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(
                                self.tupAdd(space, neighbor), self.tupAdd(space, (neighbor[0], 1))):
                            neighbors.append((neighbor[0], -1))
        return neighbors

    #determine if there's a wall at this intersection.  if offboard, returns no wall
    def isWall(self, x, y):
        if x < 0 or x > 7 or y < 0 or y > 7:
            return 0
        else:
            return self.intersections[x][y]

    #determines if there is a wall in between two squares
    def canMoveTo(self, start, end):
        if start[0] == end[0]:
            if(start[0] - 1 >= 0) and self.isWall(start[0] - 1, min(start[1], end[1])) == 2:
                return False
            if(start[0] <= 8 and self.isWall(start[0], min(start[1], end[1])) == 2):
                return False
        else:
            if(start[1] - 1 >= 0) and self.isWall(min(start[0], end[0]), start[1] - 1) == 1:
                return False
            if(start[1] <= 8 and self.isWall(min(start[0], end[0]), start[1]) == 1):
                return False
        return True

    #performs an action by specified agent
    def performAction(self, agent, action):
        
        
        if action.getType() == Action.PAWN:
            self.movePawn(agent, (action.getNewX(), action.getNewY()))
        else:
            self.wallCounts[agent] -= 1
            self.placeWall(action.getX(), action.getY())



    #moves a pawn from one square to the next
    def movePawn(self, agent, target):
        oldPos = self.agents[agent]
        self.spaces[oldPos[0]][oldPos[1]] = -1
        self.spaces[target[0]][target[1]] = agent
        self.agents[agent] = target

    #places a wall at given intersection with set orientation
    def placeWall(self, location, orientation):
        self.walls.append((location, orientation))
        self.intersections[location[0]][location[1]] = orientation

    #returns a complete list of all legal moves given player can make
    def getLegalMoves(self, agent):
        #first get legal pawn moves, designated by 'p' as first element.
        moves = []
        for item in self.getPawnMoves(self.agents[agent]):
            moves.append(('p', (item[0], item[1])))
        if self.wallCounts[agent] > 0:
            for i in range(len(self.intersections)):
                for j in range(len(self.intersections[i])):
                    if not self.isWall(i, j):
                        if not self.isWall(i, j + 1) == 1 and not self.isWall(i, j - 1) == 1:
                            #ensure that there is still a possible path
                            moves.append(('w', (i, j), 1))
                        if not self.isWall(i + 1, j) == 2 and not self.isWall(i - 1, j) == 2:
                            #ensure that there will still be a possible path
                                moves.append(('w', (i, j), 2))
        return moves

    #returns whether a certain move is legal
    def isLegalMove(self, agent, move):
        print("move: ", move)
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
        return AStar(self, space, lambda square : square[1] == edge, lambda square : abs(square[1] - edge))[0] != -1