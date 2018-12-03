import pygame
from AStar import AStar

import copy
from Agents import Action

class Qoridor:
    def __init__(self, screen):
        self.spaces = [[-1 for x in range(9)] for x in range(9)]
        self.intersections = [[0 for x in range(8)] for x in range(8)]
        self.agents = [(4, 0), (4, 8)]
        for i in range(len(self.agents)):
            self.spaces[self.agents[i][0]][self.agents[i][1]] = i
        self.agentColors = [(0, 0, 255), (255, 0, 0)]
        self.walls = []
        for i in self.walls: #for debugging
            self.intersections[i[0]][i[1]] = i[2]
        self.wallCounts = [10, 10]
        self.turn = 1
        self.screen = screen
        self.squareColor = (0, 255, 0)
        self.wallColor = (101, 67, 33)
        self.lastMaybeMove = ('p', -1, -1)



    def getGridSize(self):
        return 9

    #returns winner if game is over, else returns -1
    def endGame(self):
        if self.agents[0][1] == 8:
            return 0
        elif self.agents[1][1] == 0:
            return 1
        else:
            return -1

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
    def draw(self, currAgent, mousePos):

        #needs to be edited to only display changes.

        boxSize = self.screen.get_width()/9
        shift = (self.screen.get_width() - boxSize*9)/2
        for i in range(9):
            for j in range(9):
                pygame.draw.rect(self.screen, self.squareColor, [i*boxSize + shift, j*boxSize + shift, boxSize, boxSize], 10)
        for i in range(len(self.agents)):
            pygame.draw.circle(self.screen, self.agentColors[i], (int(self.agents[i][0] * boxSize + boxSize/2 + shift), int(self.agents[i][1] * boxSize + boxSize/2 + shift)), int(boxSize * .25))
        for i in self.walls:
            #if __name__ == '__main__':
                if i[1] == 2:
                    pygame.draw.rect(self.screen, self.wallColor, [i[0][0]*boxSize + shift + 5, (i[0][1] + 1)*boxSize + shift, boxSize*2 - 10, 1], 10)
                else:
                    pygame.draw.rect(self.screen, self.wallColor, [(i[0][0] + 1) * boxSize + shift, i[0][1] * boxSize + shift + 5, 1, boxSize*2 - 10], 10)

        #show potential move
        self.lastMaybeMove = maybeMove = self.getMoveFromMousePos(currAgent, mousePos)
        if self.isLegalMove(currAgent, maybeMove):
            s = pygame.Surface((self.screen.get_height(), self.screen.get_width()), pygame.SRCALPHA)
            if maybeMove[0] == 'p':
                pass
                pygame.draw.circle(s, self.agentColors[currAgent] + (128,), (
                maybeMove[1][0] * boxSize + boxSize / 2 + shift, maybeMove[1][1] * boxSize + boxSize / 2 + shift),
                                   int(boxSize * .25))
            else:
                if maybeMove[2] == 2:
                    pygame.draw.rect(s, self.wallColor + (255,),
                                     [maybeMove[1][0] * boxSize + shift + 5, (maybeMove[1][1] + 1) * boxSize + shift, boxSize * 2 - 10, 1],
                                     10)
                else:
                    pygame.draw.rect(s, self.wallColor + (255,),
                                     [(maybeMove[1][0] + 1) * boxSize + shift, maybeMove[1][1] * boxSize + shift + 5, 1, boxSize * 2 - 10],
                                     10)
            self.screen.blit(s, (0, 0))

    #gets value of a space, returns 7 if out of bounds
    def getSpace(self, space):
        if space[0] >= 0 and space[0] <= 8 and space[1] >= 0 and space[1] <= 8:
            return self.spaces[space[0]][space[1]]
        else:
            return 7;

    #adds 2d tuples
    def tupAdd(self, tup1, tup2):
        return (tup1[0] + tup2[0], tup1[1] + tup2[1])

    def maybeMoveChanged(self, agent, mousePosition):
        maybeMove = self.getMoveFromMousePos(agent, mousePosition)
        test = (maybeMove != self.lastMaybeMove)
        return maybeMove != self.lastMaybeMove


    #gets all possible pawn moves as absolute positions
    def getPawnMoves(self, space):
        #print "getting pawn moves ", space
        neighbors = [space]
        for neighbor in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target = self.tupAdd(space, neighbor)
            if self.getSpace(target) == -1:
                if self.canMoveTo(space, target):
                    neighbors.append(target)
            #if no direct move available, see if we can jump the pawn directly
            elif self.getSpace(self.tupAdd(space, [2*x for x in neighbor])) == -1 and self.canMoveTo(target, self.tupAdd(space, [2*x for x in neighbor])):
                neighbors.append(self.tupAdd(space, tuple(2*x for x in neighbor)))
            else: #now check diagonal jumps.
                if neighbor[0] == 0:
                    diag = self.tupAdd(space, (1, neighbor[1]))
                    if self.getSpace(diag) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(self.tupAdd(space, neighbor), diag):
                            neighbors.append(diag)
                    diag = self.tupAdd(space, (-1, neighbor[1]))
                    if self.getSpace(diag) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(self.tupAdd(space, neighbor), diag):
                            neighbors.append(diag)
                else:
                    diag = self.tupAdd(space, (neighbor[0], 1))
                    if self.getSpace(diag) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(
                                self.tupAdd(space, neighbor), diag):
                            neighbors.append(diag)
                    diag = self.tupAdd(space, (neighbor[0], -1))
                    if self.getSpace(diag) == -1:
                        if self.canMoveTo(space, self.tupAdd(space, neighbor)) and self.canMoveTo(
                                self.tupAdd(space, neighbor), diag):
                            neighbors.append(diag)
        return neighbors

    #determine if there's a wall at this intersection.  if offboard, returns no wall
    def isWall(self, x, y):
        if x < 0 or x > 7 or y < 0 or y > 7:
            return 0
        else:
            return self.intersections[x][y]

    #determines if there is a wall in between two square locations
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

    #performs an action by specified agent.  absolute positions
    def performAction(self, agent, action):
        print("agent: ", agent)
        print("action: ", action)
        '''
        if action == 'p':
            action = Action(Action.PAWN, None, action[1][0], action[1][2])
        elif action == 'w':
            action = Action(Action.WALL, None, action[1][0], action[1][2])
        '''

        if action.getType() == Action.PAWN:
            print("moving")
            self.movePawn(agent, (action.getNewX(), action.getNewY()))
        else:
            self.wallCounts[agent] -= 1
            print("placing a wall")
            if action.getOrientation() == Action.VERTICAL:
                orientation = 1
            elif action.getOrientation() == Action.HORIZONTAL:
                orientation = 2
                
            self.placeWall((action.getX(), action.getY()), orientation)

    def playerAction(self, agent, mousePosition):
        #determine location of mouse in board
        move = self.getMoveFromMousePos(agent, mousePosition)
        if self.isLegalMove(agent, move):
            self.performAction(agent, move)
            return True
        else:
            return False

    def getMoveFromMousePos(self, agent, mousePosition):
        color = self.screen.get_at(mousePosition)
        if color != self.wallColor and color != self.squareColor :
            xCoord = round(mousePosition[0] * 9 / self.screen.get_width())
            yCoord = round(mousePosition[1]*9 / self.screen.get_height())
            if xCoord < 0:
                xCoord = 0
            if yCoord < 0:
                yCoord = 0
            if xCoord > 8:
                xCoord = 8
            if yCoord > 8:
                yCoord = 8
            return ('p', (xCoord, yCoord))
        else:
            boxSize = self.screen.get_width() / 9;
            xCoord = round((mousePosition[0] - boxSize / 2) * 9 / self.screen.get_width())
            yCoord = round((mousePosition[1] - boxSize / 2) * 9 / self.screen.get_height())
            if xCoord < 0:
                xCoord = 0
            if yCoord < 0:
                yCoord = 0
            if xCoord > 7:
                xCoord = 7
            if yCoord > 7:
                yCoord = 7

            # determine orientation
            # check if valid
            # GO GO GO GO GO

            # determine location of target intersection
            actualLocation = ((xCoord + 1) * self.screen.get_width() / 9, (yCoord + 1) * self.screen.get_width() / 9);
            if (abs(mousePosition[0] - actualLocation[0]) > abs(mousePosition[1] - actualLocation[1])):
                orientation = 2
            else:
                orientation = 1
            return ('w', (xCoord, yCoord), orientation)


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
    #would be faster to return all walls, and then when they try a move, first check if it is valid.
    def getLegalMoves(self, agent):
        #first get legal pawn moves, designated by 'p' as first element.
        moves = []
        for item in self.getPawnMoves(self.agents[agent]):
            moves.append(('p', (item [0], item[1])))
        if self.wallCounts[agent] > 0:
            for i in range(len(self.intersections)):
                for j in range(len(self.intersections[i])):
                    if not self.isWall(i, j):
                        if not self.isWall(i, j + 1) == 1 and not self.isWall(i, j - 1) == 1:
                            #ensure that there is still a possible path
                            self.placeWall((i, j), 1)
                            if self.pathExists(self.agents[0], 8) and self.pathExists(self.agents[1], 0):
                                moves.append(('w', (i, j), 1))
                            self.removeWall((i, j))
                        if not self.isWall(i + 1, j) == 2 and not self.isWall(i - 1, j) == 2:
                            #ensure that there will still be a possible path
                            self.placeWall((i, j), 2)
                            if self.pathExists(self.agents[0], 8) and self.pathExists(self.agents[1], 0):
                                moves.append(('w', (i, j), 2))
                            self.removeWall((i, j))
        return moves

    def removeWall(self, location):
        self.walls.pop()
        self.intersections[location[0]][location[1]] = 0

    #returns whether a certain move is legal. Takes absolutes
    def isLegalMove(self, agent, move):
        #print("move: ", move)
        if move[0] == 'p':
            legalMoves = self.getPawnMoves(self.agents[agent])
            for i in legalMoves:
                if(i[0] == move[1][0] and i[1] == move[1][1]):
                    return True
            return False
        else:
            if(self.wallCounts[agent] == 0):
                return False
            if(self.intersections[move[1][0]][move[1][1]]) != 0:
                return False
            self.placeWall(move[1], move[2])
            if not self.pathExists(self.agents[0], 8) or not self.pathExists(self.agents[1], 0):
                self.removeWall(move[1])
                return False
            self.removeWall(move[1])
            if not self.isWall(move[1][0], move[1][1]):
                if move[2] == 1 and not self.isWall(move[1][0], move[1][1] + 1) == 1 and not self.isWall(move[1][0], move[1][1] - 1) == 1:
                    return True
                if move[2] == 2 and not self.isWall(move[1][0] + 1, move[1][1]) == 2 and not self.isWall(move[1][0] - 1, move[1][1]) == 2:
                    return True
        return False

    #edge is 0 or 8 to indicate which edge
    def pathExists(self, space, edge):
        return AStar(self, space, lambda square : square[1] == edge, lambda square : abs(square[1] - edge))[0] != -1
