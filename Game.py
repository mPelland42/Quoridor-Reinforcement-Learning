import pygame
from pygame.locals import *

from AStar import AStar

import time

import sys

import random

from Agents import Action
from GameState import GameState
from GameState import BoardElement
from Point import Point

from Agents import TopAgent
from Agents import BottomAgent

import math

pygame.init()



REWARD_WIN = 1.5
REWARD_LOSE = -1.5
REWARD_ILLEGAL = -0.75
REWARD_GOOD_DIRECTION = 0.02
REWARD_BAD_DIRECTION = -0.04
REWARD_GOOD_WALL = 0.02
REWARD_BAD_WALL = -0.04



SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400



class Qoridor:
    def __init__(self, gridSize, numWalls, gameSpeedSlow, startWithDrawing, humanPlaying):

        # game colors for display
        self.agentColors = [(230, 46, 0), (0, 0, 255)] #red & blue
        self.squareColor = (255, 255, 255) # light green
        self.wallColor = (51, 153, 51) # green

        # flags
        self.gridSize = gridSize
        self.numWalls = numWalls
        self.gameSpeedSlow = gameSpeedSlow
        self.gameSpeed = 0
        self.humanPlaying = humanPlaying


        self.actions = Action.makeAllActions(gridSize)

        self.movesTillVictory = []

        self.initialDraw = True
        self.currentlyDrawing = startWithDrawing
        self.learning = not humanPlaying
        
        self.printStuff = False
        self.printQ = False
                        
        print("Learning: ", self.learning)
        print("drawing: ", self.currentlyDrawing)
        print("game speed: ", self.gameSpeed)
        print("printing: ", self.printStuff)
    
        
        # reset game state
        self.reset()

        self.localAvgGameLength = 0
        self.games = 0
        self.victories = {BoardElement.AGENT_TOP: 0, BoardElement.AGENT_BOT: 0}




    def setLearningParameters(self, sess, model, memory, MAX_EPSILON, MIN_EPSILON, LAMBDA):
        self.sess = sess
        self.model = model
        self.memory = memory
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA

        self.epsilon = MAX_EPSILON
        self.steps = 0
        self.recentReward = 0
        self.rewardStore = []
        self.MaxXStore = []

        print("Setting up agent networks...")
        self.topAgent = TopAgent(self, sess, model, memory, REWARD_ILLEGAL)
        self.bottomAgent = BottomAgent(self, sess, model, memory, REWARD_ILLEGAL)
        self.agents = [self.bottomAgent, self.topAgent]
        print("completed\n")


    def reset(self):
        self.lastMaybeMove = None

        self.movesTaken = 0
        self.visited = []
        self.gameReward = 0
        self.state = GameState(self.gridSize, self.numWalls)
        
        # also reset the visuals
        if self.currentlyDrawing or self.initialDraw:
            self.initialDraw = False
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)
            self.draw(0, (0, 0))
            pygame.display.flip()
            pygame.display.update()
            self.initialDraw = False
            


    def run(self):

        # reset game
        self.reset()
        firstMove = True

        previousState = None
        previousActionIndex = None
        previousReward = None
        newState = None
        previousNewState =  None
        done = False
        
        currentAgent = random.randint(0, 1)


        # run game till we have a winner
        while not done:
            #pygame handling
            
            #drawn = False
            if not self.humanPlaying:
                #print ("\n============================================")

                agent = self.agents[currentAgent]
                agentType = agent.getType()

                state = self.state.asVector(agentType)
                
                
                
                if (self.printStuff):
                    if agentType == BoardElement.AGENT_TOP:
                        print("agent top")
                    if agentType == BoardElement.AGENT_BOT:
                        print("agent bot")
                

                actionIndex, action = agent.move(self.epsilon, state)
                self.movesTaken += 1
                
                reward = self.performAction(agentType, action)
                    
                #reward += -self.state.getMovesTaken()
                newState = self.state.asVector(agentType)

                if self.printStuff:
                    print("reward: ", reward)
                    print(" ")
                    
                if self.learning:
                    if not firstMove:
                        if self.state.getWinner() == None:
                            self.memory.addSample((previousState, previousActionIndex, previousReward, previousNewState))
                            
                        else:
                            self.memory.addSample((state, actionIndex, REWARD_WIN, None))
                            self.memory.addSample((previousState, previousActionIndex, REWARD_LOSE, previousNewState))
                            self.victories[agentType] += 1
                            done = True
                            
                            
                            
                        self.gameReward += reward
                        self.recentReward += reward
                        
                        agent.learn()
                    else:
                        firstMove = False
                        
                    self.steps += 1
                    self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) \
                        * math.exp(-self.LAMBDA * self.steps)

                if not self.state.getWinner() == None:
                    done = True

                previousState = state
                previousActionIndex = actionIndex
                previousReward = reward
                previousNewState = newState
                currentAgent = (currentAgent + 1) % 2
                


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        # this just displays the games
                        self.currentlyDrawing = not self.currentlyDrawing
                        print ("currentlyDrawing: ", self.currentlyDrawing)
                        
                        #this toggles between instant drawing (which is cool to see) and
                        # a more practical slow drawing to inspect the agent's moves
                    elif event.key == pygame.K_f:
                        if self.gameSpeed == self.gameSpeedSlow:
                            self.gameSpeed = 0
                        else:
                            self.gameSpeed = self.gameSpeedSlow
                            #can only see effects if you are currently drawing
                        print("switched to gamespeed to ", self.gameSpeed)
                            
                    elif event.key == pygame.K_l:
                        # turning learning off, will cause all actions taken to be a prediction
                        # from the model, so no randomness is involved, it also
                        # stops updating actins/reward pairs to memory and turns learning off 
                        # until this is turned back on. It's not good to learn without randomness basically
                        
                        self.learning = not self.learning
                        print ("learning: ", self.learning)
                        
                        
                    elif event.key == pygame.K_s:
                        print("save functionality of the model has not been implemented yet")
                        self.topAgent.saveState()
                        self.bottomAgent.saveState()
                        
                        
                        
                    elif event.key == pygame.K_p:
                        self.printStuff = not self.printStuff
                        print("printing: ", self.printStuff)
                        
                    elif event.key == pygame.K_q:
                        self.printQ = True
                '''
                if event.type == pygame.MOUSEBUTTONDOWN and agents[currentAgent] == HUMAN:
                    if self.playerAction(currentAgent, pygame.mouse.get_pos()):
                        if self.endGame() != -1:
                            currentAgent = 0
                            self.reset()
                        else:
                            currentAgent = (currentAgent+1)%2
                        #self.draw(currentAgent, pygame.mouse.get_pos())
                        #drawn = True
                '''
            if(self.currentlyDrawing):
                self.draw(currentAgent, pygame.mouse.get_pos())
                pygame.display.flip()
                pygame.display.update()
                time.sleep(self.gameSpeed)


            #if self.maybeMoveChanged(currentAgent, pygame.mouse.get_pos()):
            #    self.draw(currentAgent, pygame.mouse.get_pos())

        # since they use the same model
        self.movesTillVictory.append(self.movesTaken)
        self.games += 1
        agent.getLoss()
        #print(" ", self.movesTaken, agent.getLoss())
        self.localAvgGameLength += self.movesTaken
        self.topAgent.saveState()
        self.bottomAgent.saveState()


        
        
    def reward(self, agentType, action, pathShorter, pathBlocked):
        
        
        if action.getType() == Action.PAWN:
            if pathShorter:
                return REWARD_GOOD_DIRECTION
            else:
                return REWARD_BAD_DIRECTION
        else:
            if pathBlocked:
                return REWARD_GOOD_WALL
            else:
                return REWARD_BAD_WALL
            
            
            
            
    def getLearning(self):
        return self.learning
    
    def printDetails(self, gamesPerEpoch):
        self.localAvgGameLength = self.localAvgGameLength / gamesPerEpoch
        self.recentRewardAvg = self.recentReward / gamesPerEpoch
        self.recentReward = 0
        print("Top Victories: ", self.victories[BoardElement.AGENT_TOP])
        print("Bot Victories: ", self.victories[BoardElement.AGENT_BOT])
        print("Local Average Game Length: ", self.localAvgGameLength)
        print("Local Average Game Reward: ", self.recentRewardAvg)
        print("Local Average Loss: ", self.agents[0].getRecentLoss())
        
        print("Epsilon: "+"{:.6f}".format(self.epsilon))

        self.localAvgGameLength = 0
       # print("\nMoves/loss: ")


    #performs an action by specified agent
    def performAction(self, agentType, action):
        
        pathShorter = False
        wallBlocked = False
        
        
        if action.getType() == Action.PAWN:
            
            
            if agentType == BoardElement.AGENT_TOP:
                pathLengthPrevious = AStar(self, self.state.getPosition(BoardElement.AGENT_TOP), lambda square : \
                        square.Y == self.gridSize-1, lambda square : abs(square.Y - self.gridSize-1))[2]
        
                self.state.updateAgentPosition(BoardElement.AGENT_TOP, action.getPosition())
                
                pathLengthPost = AStar(self, self.state.getPosition(BoardElement.AGENT_TOP), lambda square : \
                        square.Y == self.gridSize-1, lambda square : abs(square.Y - self.gridSize-1))[2]

                
            elif agentType == BoardElement.AGENT_BOT:
                pathLengthPrevious = AStar(self, self.state.getPosition(BoardElement.AGENT_BOT), lambda square : \
                        square.Y == 0, lambda square : abs(square.Y))[2]
                
                self.state.updateAgentPosition(BoardElement.AGENT_BOT, action.getPosition())
                
                pathLengthPost = AStar(self, self.state.getPosition(BoardElement.AGENT_BOT), lambda square : \
                        square.Y == 0, lambda square : abs(square.Y))[2]
                
            if pathLengthPost < pathLengthPrevious:
                pathShorter = True
                
        else: # wall action
            
            if agentType == BoardElement.AGENT_TOP:
                enemyPreviousPathLength =  AStar(self, self.state.getPosition(BoardElement.AGENT_BOT), lambda square : \
                square.Y == 0, lambda square : abs(square.Y))[2]
                
                self.state.addIntersection(action.getPosition(), action.getOrientation())
                self.state.removeWallCount(BoardElement.AGENT_TOP)
                
                enemyCurrentPathLength =  AStar(self, self.state.getPosition(BoardElement.AGENT_BOT), lambda square : \
                square.Y == 0, lambda square : abs(square.Y))[2]
                
            else:
                enemyPreviousPathLength = AStar(self, self.state.getPosition(BoardElement.AGENT_TOP), lambda square : \
                square.Y == self.gridSize-1, lambda square : abs(square.Y - self.gridSize-1))[2]
                
                self.state.addIntersection(action.getPosition(), action.getOrientation())
                self.state.removeWallCount(BoardElement.AGENT_BOT)
                
                enemyCurrentPathLength = AStar(self, self.state.getPosition(BoardElement.AGENT_TOP), lambda square : \
                square.Y == self.gridSize-1, lambda square : abs(square.Y - self.gridSize-1))[2]
            
            if enemyCurrentPathLength > enemyPreviousPathLength:
                wallBlocked = True
                
            
        return self.reward(agentType, action, pathShorter, wallBlocked)

    
    
    
    def getStateSize(self):
        return len(self.state.asVector(BoardElement.AGENT_TOP))

    def getActionSize(self):
        return len(Action.makeAllActions(self.gridSize))

    def getState(self):
        return self.state

    def getGridSize(self):
        return self.gridSize

    #returns winner if game is over, else returns -1
    #I don't think this is used at all anymore,
    #DEPRECATED
    def endGame(self):
        if self.agents[0][1] == self.gridSize-1:
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
        if space[0] < self.gridSize-1:
            neighbors.append((space[0] + 1, space[1]))
        if space[1] > 0:
            neighbors.append((space[0], space[1] - 1))
        if space[1] < self.gridSize-1:
            neighbors.append((space[0], space[1] + 1))
            
        return neighbors

    #draws it to the screen
    #this needs a _lot_ of edits to make up to date
    def draw(self, currAgent, mousePos):
        
        self.screen.fill(0)

        #needs to be edited to only display changes.

        boxSize = int(self.screen.get_width()/self.gridSize)
        shift = int((self.screen.get_width() - boxSize*self.gridSize)/2)
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                pygame.draw.rect(self.screen, self.squareColor, [i*boxSize + shift, j*boxSize + shift, boxSize, boxSize], 10)
        topAgentPos = self.state.agentPositions[BoardElement.AGENT_TOP]
        pygame.draw.circle(self.screen, self.agentColors[0], (int(topAgentPos.X * boxSize + boxSize/2 + shift), int(topAgentPos.Y * boxSize + boxSize/2 + shift)), int(boxSize * .25))
        botAgentPos = self.state.agentPositions[BoardElement.AGENT_BOT]
        pygame.draw.circle(self.screen, self.agentColors[1], (int(botAgentPos.X * boxSize + boxSize/2 + shift), int(botAgentPos.Y * boxSize + boxSize/2 + shift)), int(boxSize * .25))

        #needs to be rewritten/readded to how things are structured.
        for i in self.state.wallPositions:
            #if __name__ == '__main__':
                if i[1] == BoardElement.WALL_HORIZONTAL:
                    pygame.draw.rect(self.screen, self.wallColor, [i[0].X*boxSize + shift + 5, (i[0].Y + 1)*boxSize + shift, boxSize*2 - 10, 1], 10)
                else:
                    pygame.draw.rect(self.screen, self.wallColor, [(i[0].X + 1) * boxSize + shift, i[0].Y * boxSize + shift + 5, 1, boxSize*2 - 10], 10)

        #show potential move
        #currently commenting out to focus on bots
        """
        self.lastMaybeMove = maybeMove = self.getMoveFromMousePos(currAgent, mousePos)
        if self.isLegalMove(currAgent, maybeMove):
            s = pygame.Surface((self.screen.get_height(), self.screen.get_width()), pygame.SRCALPHA)
            if maybeMove[0] == 'p':
                pass
                pygame.draw.circle(s, self.agentColors[currAgent] + (128,), (
                int(maybeMove[1][0] * boxSize + boxSize / 2 + shift), int(maybeMove[1][1] * boxSize + boxSize / 2 + shift)), int(boxSize * .25))
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
            """

    #gets value of a space, returns 7 if out of bounds
    #should be unnecssary.  Replace with an "isoccupied" function.  Check if any agent is in that space.
    #space will be passed as a point
    #should be completely unneessary
    def getSpace(self, space):
        if space.X >= 0 and space.X <= self.gridSize and space.Y >= 0 and space.Y <= self.gridSize:
            if(self.state.agentPositions[BoardElement.AGENT_TOP] == space):
                return BoardElement.AGENT_TOP
            elif(self.state.agentPositions[BoardElement.AGENT_BOT] == space):
                return BoardElement.AGENT_BOT
            else:
                return BoardElement.EMPTY
        else:
            return BoardElement.OFF_GRID;

    def isClear(self, space):
        if space.X >= 0 and space.X <= self.gridSize and space.Y >= 0 and space.Y <= self.gridSize:
            return self.state.agentPositions[BoardElement.AGENT_TOP] != space and self.state.agentPositions[BoardElement.AGENT_BOT] != space
        else: #is off the grid, hence not clear
            return False

    #adds 2d tuples
    #should be completely unnecessary from here on out
    #use point + point instead
    def tupAdd(self, tup1, tup2):
        return (tup1[0] + tup2[0], tup1[1] + tup2[1])

    #maybeMove needs to be changed to be an action.
    #actions have been updated to allow for comparison
    def maybeMoveChanged(self, agent, mousePosition):
        maybeMove = self.getMoveFromMousePos(agent, mousePosition)
        #test = (maybeMove != self.lastMaybeMove)
        return maybeMove != self.lastMaybeMove

    #accepts an action and returns a point delta associated
    def actionToDelta(self, action):
        if(action == Action.UP):
            return Point(0, -1)
        elif action == Action.DOWN:
            return Point(0, 1)
        elif action == Action.RIGHT:
            return Point(1, 0)
        elif action == Action.LEFT:
            return Point(-1, 0)
        return Point(0,0)


    #gets all possible pawn moves as action objects
    def getPawnMoves(self, space):

        #print "getting pawn moves ", space
        moves = [space]
        for action in [Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)]:
            target = space + action
            if self.isClear(target):
                if self.canMoveTo(space, target):
                    moves.append(target)
            elif self.isClear(target+action) and self.canMoveTo(space, target) and self.canMoveTo(target, target+action):
                moves.append(target+action)
                
                """
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
                            neighbors.append(diag)"""
        return moves

    #determine if there's a wall at this intersection.  if offboard, returns no wall
    #this really shouldn't need to exist as it is.
    def isWall(self, x, y):
        if x < 0 or x > self.gridSize-2 or y < 0 or y > self.gridSize-2:
            return 0
        else:
            return self.state.intersections[x][y]

    #determines if there is a wall in between two adjacent square locations
    def canMoveTo(self, start, end):
        if end.X < 0 or end.X >= self.getGridSize() or end.Y < 0 or end.Y >= self.getGridSize():
            return False
        if (abs(end.X - start.X + end.Y-start.Y) != 1):
            #provided cells are not adjacent.
            return False

        if start.X == end.X:
            if(start.X - 1 >= 0) and self.isWall(start.X - 1, min(start.Y, end.Y)) == BoardElement.WALL_HORIZONTAL:
                return False
            if(start.X <= self.gridSize-1 and self.isWall(start.X, min(start.Y, end.Y)) == BoardElement.WALL_HORIZONTAL):
                return False
        else:
            if(start.Y - 1 >= 0) and self.isWall(min(start.X, end.X), start.Y - 1) == BoardElement.WALL_VERTICAL:
                return False
            if(start.Y <= self.gridSize-1 and self.isWall(min(start.X, end.X), start.Y) == BoardElement.WALL_VERTICAL):
                return False
        return True



        
    def playerAction(self, agent, mousePosition):
        #determine location of mouse in board
        move = self.getMoveFromMousePos(agent, mousePosition)
        if self.isLegalMove(agent, move):
            self.performAction(agent, move)
            return True
        else:
            return False

    #this needs some hefty work to fic up.
    #low priority, focus on AI functions for now
    #agent is one of BoardElement.AGENT_TOP or AGENT_BOTTOM
    def getMoveFromMousePos(self, agent, mousePosition):
        color = self.screen.get_at(mousePosition)
        if color != self.wallColor and color != self.squareColor :
            xCoord = int(mousePosition[0] * (self.gridSize) / self.screen.get_width())
            yCoord = int(mousePosition[1] * (self.gridSize) / self.screen.get_height())
            if xCoord < 0:
                xCoord = 0
            if yCoord < 0:
                yCoord = 0
            if xCoord > self.gridSize-1:
                xCoord = self.gridSize-1
            if yCoord > self.gridSize-1:
                yCoord = self.gridSize-1
            return ('p', (xCoord, yCoord))
        else:
            boxSize = self.screen.get_width() / (self.gridSize);
            xCoord = int((mousePosition[0] - boxSize / 2) * (self.gridSize) / self.screen.get_width())
            yCoord = int((mousePosition[1] - boxSize / 2) * (self.gridSize) / self.screen.get_height())
            if xCoord < 0:
                xCoord = 0
            if yCoord < 0:
                yCoord = 0
            if xCoord > (self.gridSize-2):
                xCoord = (self.gridSize-2)
            if yCoord > (self.gridSize-2):
                yCoord = (self.gridSize-2)

            # determine orientation
            # check if valid
            # GO GO GO GO GO

            # determine location of target intersection
            actualLocation = ((xCoord + 1) * self.screen.get_width() / (self.gridSize), (yCoord + 1) * self.screen.get_width() / (self.gridSize));
            if (abs(mousePosition[0] - actualLocation[0]) > abs(mousePosition[1] - actualLocation[1])):
                orientation = BoardElement.WALL_HORIZONTAL
            else:
                orientation = BoardElement.WALL_VERTICAL
            return ('w', (xCoord, yCoord), orientation)


    #moves a pawn from one square to the next
    #DEPRECATED, should literally never be called anymore.
    def movePawn(self, agent, target):
        #print("movePawn()")
        #print(agent)
        #print(target)
        #rint("\n")
        oldPos = self.agents[agent]
        self.spaces[oldPos[0]][oldPos[1]] = -1
        self.spaces[target[0]][target[1]] = agent
        self.agents[agent] = target



    #Use instead of addIntersection when only temporarily adding a wall
    #in order to check legality of a move
    def placeTempWall(self, location, orientation):
        #self.walls.append((location, orientation))
        self.state.intersections[location.X][location.Y] = orientation

    #returns a complete list of all legal moves given player can make
    #would be faster to return all walls, and then when they try a move, first check if it is valid.
    #this needs to _really_ be fixed
    #DEPRECATED DO NOT USE
    '''
    def getLegalActions(self, agent):
        #first get legal pawn actions
        moves = self.getPawnMoves(self.getAgentPosition(agent))
        if self.wallCounts[agent] > 0:
            for i in range(len(self.intersections)):
                for j in range(len(self.intersections[i])):
                    if not self.isWall(i, j):
                        if not self.isWall(i, j + 1) == 1 and not self.isWall(i, j - 1) == 1:
                            #ensure that there is still a possible path
                            self.placeWall((i, j), 1)
                            if self.pathExists(self.agents[0], self.gridSize-1) and self.pathExists(self.agents[1], 0):
                                moves.append(('w', (i, j), 1))
                            self.removeWall((i, j))
                        if not self.isWall(i + 1, j) == 2 and not self.isWall(i - 1, j) == 2:
                            #ensure that there will still be a possible path
                            self.placeWall((i, j), 2)
                            if self.pathExists(self.agents[0], self.gridSize-1) and self.pathExists(self.agents[1], 0):
                                moves.append(('w', (i, j), 2))
                            self.removeWall((i, j))
        return moves
    '''

    #swap to using gameState object
    #used to undo temp placement for testing of validity
    def removeTempWall(self, location):
        #self.walls.pop()
        self.state.intersections[location.X][location.Y] = 0

    #returns whether a certain move is legal. Takes absolutes
    #TAG
    def isLegalMove(self, agent, move):
        #print("isLegalMove() move: ", move)
        if move.getType() == Action.PAWN:
            legalMoves = self.getPawnMoves(self.state.agentPositions[agent])
            for i in legalMoves:
                #print("legalMoves: ", legalMoves)
                if(i == move.position):
                    return True
            return False
        else:
            if(self.state.walls[agent] == 0):
                return False
            #print("lol: ", self.intersections[move[1][0]][move[1][1]])
            #print("X: ", move[1][0])
            #print("Y: ", move[1][1])
            #print("intersection: ", self.intersections[move[1][0]][move[1][1]])
            if(self.state.intersections[move.position.X][move.position.Y]) != 0:
                return False
            self.placeTempWall(move.position, move.orientation)
            if not self.pathExists(self.state.agentPositions[BoardElement.AGENT_TOP], self.gridSize-1) or not self.pathExists(self.state.agentPositions[BoardElement.AGENT_BOT], 0):
                self.removeTempWall(move.position)
                return False
            self.removeTempWall(move.position)
            if not self.isWall(move.position.X, move.position.Y):
                if move.orientation == BoardElement.WALL_VERTICAL  and not self.isWall(move.position.X, move.position.Y + 1) == BoardElement.WALL_VERTICAL and not self.isWall(move.position.X, move.position.Y - 1) == BoardElement.WALL_VERTICAL:
                    return True
                if move.orientation == BoardElement.WALL_HORIZONTAL and not self.isWall(move.position.X + 1, move.position.Y) == BoardElement.WALL_HORIZONTAL and not self.isWall(move.position.X - 1, move.position.Y) == BoardElement.WALL_HORIZONTAL:
                    return True
        return False

    #edge is 0 or 8 to indicate which edge
    #TAG probably
    def pathExists(self, space, edge):
        return AStar(self, space, lambda square : square.Y == edge, lambda square : abs(square.Y - edge))[0] != -1
