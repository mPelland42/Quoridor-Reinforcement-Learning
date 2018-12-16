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
import numpy as np

from Agents import TopAgent
from Agents import BottomAgent

import math

pygame.init()


#REWARDS
REWARD_WIN = 1.0
REWARD_LOSE = -1.0

REWARD_ILLEGAL = -.50
REWARD_GOOD_DIRECTION = -.01
REWARD_BAD_DIRECTION = -.01
REWARD_GOOD_WALL = -.01
REWARD_BAD_WALL = -.01

PREVIOUS_ACTIONS_LEN = 0


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
        self.random = True
        self.predictGames = False

        self.printStuff = False
        self.printQ = False

        print("Random: ", self.random)
        print("drawing: ", self.currentlyDrawing)
        print("game speed: ", self.gameSpeed)
        print("printing: ", self.printStuff)


        # reset game state
        self.reset()

        self.localAvgGameLength = 0
        self.winPredictionLoss = 0
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
        self.topAgent = TopAgent(self, sess, model, memory, REWARD_ILLEGAL, self.humanPlaying)
        self.bottomAgent = BottomAgent(self, sess, model, memory, REWARD_ILLEGAL, self.humanPlaying)
        self.agents = [self.bottomAgent, self.topAgent]
        print("completed\n")

    #reset state after each game
    def reset(self):
        self.lastMaybeMove = None

        self.movesTaken = 0
        self.visited = []
        self.gameReward = 0
        self.state = GameState(self.gridSize, self.numWalls, PREVIOUS_ACTIONS_LEN)

        # also reset the visuals
        if self.currentlyDrawing or self.initialDraw:
            self.initialDraw = False
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)
            self.draw(0, (0, 0))
            pygame.display.flip()
            pygame.display.update()
            self.initialDraw = False


    #play the game based on set parameters
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
            if not self.humanPlaying or currentAgent == 0:
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
                if actionIndex == -1:
                    print("WHAT!?")
                    break

                self.movesTaken += 1



                #reward += -self.state.getMovesTaken()
                newState = self.state.asVector(agentType)


                reward = self.performAction(agentType, action)
                self.state.addAction(actionIndex, agentType)
                
                
                if self.printStuff:
                    print("reward: ", reward)

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


                if(self.predictGames):
                    p = self.model.predictOneProb(newState, self.sess)
                    if agentType == BoardElement.AGENT_TOP:
                        print("Predicting that top has {} chance of winning".format(p))
                    else:
                        print("Predicting that bot has {} chance of winning".format(p))
                    self.predictGames = False
                        
                
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
                            
                    elif event.key == pygame.K_r:
                        # turning random off, will cause all actions taken to be a prediction
                        # from the model, so no randomness is involved
                        self.random = not self.random
                        print ("Random: ", self.random)

                    elif event.key == pygame.K_h:
                        self.humanPlaying = not self.humanPlaying
                    elif event.key == pygame.K_s:
                        self.model.save(self.sess)
                        

                    elif event.key == pygame.K_p:
                        self.printStuff = not self.printStuff
                        print("printing: ", self.printStuff)

                    elif event.key == pygame.K_q:
                        self.printQ = True
                        
                    elif event.key == pygame.K_w:
                        self.predictGames = True
                
                if event.type == pygame.MOUSEBUTTONDOWN and self.humanPlaying and currentAgent == 1:
                    if self.playerAction(BoardElement.AGENT_TOP, pygame.mouse.get_pos()):
                        if not self.state.getWinner() == None:
                            done = True
                            self.reset()
                        else:
                            currentAgent = (currentAgent+1)%2
                        #self.draw(currentAgent, pygame.mouse.get_pos())
                        #drawn = True
                
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
        self.memory.addStateWinMemory(self.state.getStateMemory())
        
        for i in range(3):
            self.trainStateWinPrediction()
            
        agent.getLoss()
        #print(" ", self.movesTaken, agent.getLoss())
        self.localAvgGameLength += self.movesTaken



    #determine reward for a given action
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




    def getRandom(self):
        return self.random

    #display results of learning
    def printDetails(self, gamesPerEpoch):
        self.model.save(self.sess)
        self.localAvgGameLength = self.localAvgGameLength / gamesPerEpoch
        self.recentRewardAvg = self.recentReward / gamesPerEpoch
        self.recentReward = 0
        print("Top Victories: ", self.victories[BoardElement.AGENT_TOP])
        print("Bot Victories: ", self.victories[BoardElement.AGENT_BOT])
        print("Local Average Game Length: ", self.localAvgGameLength)
        print("Local Average Game Reward: ", self.recentRewardAvg)
        print("Local Average Loss: ", self.agents[0].getRecentLoss())
        print("Local Win Prediction Loss: ", self.winPredictionLoss / gamesPerEpoch)

        print("Epsilon: "+"{:.6f}".format(self.epsilon))

        self.localAvgGameLength = 0
        self.winPredictionLoss = 0
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




    def trainStateWinPrediction(self):
        batch = self.memory.sampleStateWinBatch()
        
        x = np.zeros((len(batch), self.model.getNumStates()))
        y = np.zeros((len(batch), 1))
        for i, b in enumerate(batch):
            state, win = b[0], b[1]
            
            x[i] = state
            y[i] = win
            
        _, l = self.model.trainBatchProb(self.sess, x, y)
        self.winPredictionLoss += l


    def getStateSize(self):
        return len(self.state.asVector(BoardElement.AGENT_TOP))

    def getActionSize(self):
        return len(Action.makeAllActions(self.gridSize))

    def getState(self):
        return self.state

    def getGridSize(self):
        return self.gridSize
    
    #draw the current gamestate to the screen
    def draw(self, currAgent, mousePos):

        self.screen.fill(0)

        #needs to be edited to only display changes.

        boxSize = int(self.screen.get_width()/self.gridSize)
        shift = int((self.screen.get_width() - boxSize*self.gridSize)/2)
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                pygame.draw.rect(self.screen, self.squareColor, [i*boxSize + shift, j*boxSize + shift, boxSize, boxSize], 10)
        topAgentPos = self.state.getPosition(BoardElement.AGENT_TOP)
        pygame.draw.circle(self.screen, self.agentColors[0], (int(topAgentPos.X * boxSize + boxSize/2 + shift), int(topAgentPos.Y * boxSize + boxSize/2 + shift)), int(boxSize * .25))
        botAgentPos = self.state.getPosition(BoardElement.AGENT_BOT)
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
    #returns if a specific space in grid is unoccupied
    def isClear(self, space):
        if space.X >= 0 and space.X <= self.gridSize and space.Y >= 0 and space.Y <= self.gridSize:
            return self.state.getPosition(BoardElement.AGENT_TOP) != space and self.state.getPosition(BoardElement.AGENT_BOT) != space
        else: #is off the grid, hence not clear
            return False



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
        moves = []
        for action in [Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)]:
            target = space + action
            if self.isClear(target):
                if self.canMoveTo(space, target):
                    moves.append(target)
            elif self.isClear(target+action) and self.canMoveTo(space, target) and self.canMoveTo(target, target+action):
                moves.append(target+action)

            #diagonal jump checks deprecated and unused
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
    def whatWall(self, x, y):
        if x < 0 or x > self.gridSize-2 or y < 0 or y > self.gridSize-2:
            return BoardElement.OFF_GRID
        else:
            return self.state.getIntersection(Point(x,y))

    #determines if there is a wall in between two adjacent square locations
    def canMoveTo(self, start, end):
        if end.X < 0 or end.X >= self.getGridSize() or end.Y < 0 or end.Y >= self.getGridSize():
            return False
        if (abs(end.X - start.X + end.Y-start.Y) != 1):
            #provided cells are not adjacent.
            return False
        if start.X == end.X:
            if(start.X - 1 >= 0) and (self.whatWall(start.X - 1, min(start.Y, end.Y)) == BoardElement.WALL_HORIZONTAL):
                return False
            if(start.X <= self.gridSize-1) and (self.whatWall(start.X, min(start.Y, end.Y)) == BoardElement.WALL_HORIZONTAL):
                return False
        else:
            if(start.Y - 1 >= 0) and (self.whatWall(min(start.X, end.X), start.Y - 1) == BoardElement.WALL_VERTICAL):
                return False
            if(start.Y <= self.gridSize-1) and (self.whatWall(min(start.X, end.X), start.Y) == BoardElement.WALL_VERTICAL):
                return False
        return True



    #perform human action based on mouse click
    def playerAction(self, agent, mousePosition):
        #determine location of mouse in board
        move = self.getMoveFromMousePos(agent, mousePosition)
        if self.isLegalMove(agent, move):
            self.performAction(agent, move)
            return True
        else:
            return False

    #determine the move human player is trying to make bsaed on mouse coords.
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
            return Action(Action.PAWN, None, None, Point(xCoord, yCoord))
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

            # determine location of target intersection
            actualLocation = ((xCoord + 1) * self.screen.get_width() / (self.gridSize), (yCoord + 1) * self.screen.get_width() / (self.gridSize));
            if (abs(mousePosition[0] - actualLocation[0]) > abs(mousePosition[1] - actualLocation[1])):
                orientation = BoardElement.WALL_HORIZONTAL
            else:
                orientation = BoardElement.WALL_VERTICAL
            return Action(Action.WALL, None, orientation, Point(xCoord, yCoord))

    #Use instead of addIntersection when only temporarily adding a wall
    #in order to check legality of a move
    def placeTempWall(self, location, orientation):
        #self.walls.append((location, orientation))
        self.state.intersections[location.X][location.Y] = orientation

    #used to undo temp placement for testing of validity
    def removeTempWall(self, location):
        #self.walls.pop()
        self.state.intersections[location.X][location.Y] = BoardElement.EMPTY

    #returns whether a certain move is legal.
    def isLegalMove(self, agent, move):
        if move.getType() == Action.PAWN:
            legalMoves = self.getPawnMoves(self.state.getPosition(agent))
            for i in legalMoves:
                #print("legalMoves: ", legalMoves)
                if(i == move.position):
                    return True
            return False
        else:
            if self.state.getWallCount(agent) == 0:
                return False
            
            if self.state.getIntersection(move.getPosition()) != BoardElement.EMPTY:
                return False
            
            self.placeTempWall(move.position, move.orientation)
            if not self.pathExists(self.state.agentPositions[BoardElement.AGENT_TOP], self.gridSize-1) or not self.pathExists(self.state.agentPositions[BoardElement.AGENT_BOT], 0):
                self.removeTempWall(move.position)
                return False
            self.removeTempWall(move.position)
            
            if self.whatWall(move.position.X, move.position.Y) == BoardElement.EMPTY:
                if move.orientation == BoardElement.WALL_VERTICAL  and not (self.whatWall(move.position.X, move.position.Y + 1) == BoardElement.WALL_VERTICAL) and not (self.whatWall(move.position.X, move.position.Y - 1) == BoardElement.WALL_VERTICAL):
                    return True
                if move.orientation == BoardElement.WALL_HORIZONTAL and not (self.whatWall(move.position.X + 1, move.position.Y) == BoardElement.WALL_HORIZONTAL) and not (self.whatWall(move.position.X - 1, move.position.Y) == BoardElement.WALL_HORIZONTAL):
                    return True
        return False
    
    #determines if a path exists on the gameboard from the given space
    #   to the target edge
    def pathExists(self, space, edge):
        return AStar(self, space, lambda square : square.Y == edge, lambda square : abs(square.Y - edge))[0] != -1
