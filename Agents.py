#aconda navigator
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
import os



class Action:
    #all possible actions
    PAWN = "PAWN"
    WALL = "WALL"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    JUMP_UP = "JUMP UP"
    JUMP_DOWN = "JUMP DOWN"
    JUMP_LEFT = "JUMP LEFT"
    JUMP_RIGHT = "JUMP RIGHT"

    PAWN_MOVES = 8
    TOP_K_SELECTION = 20
    MOVE_PROBABILITY = .70
    GAMMA = 0.80
    greedy = True

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


        for y in range(gridSize - 1):
            for x in range(gridSize - 1):
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

    def __eq__(self, other):
        return self.actionType == other.actionType and self.direction == other.direction and self.orientation == other.orientation and self.position == other.position
    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return "Action: " + self.xstr(self.actionType) + " Direction: " + self.xstr(self.direction) + \
    " Orientation: " + self.xstr(self.orientation) + "  X,Y:" + str(self.position)

    def __repr__(self):
        return self.__str__()




# self.game should be automatically updated when changes outside of this scope,
# (passed as a deep copy)
class Agent:
    def __init__(self, game, agentType, sess, model, memory, rewardInvalid):
        self.game = game
        self.sess = sess
        self.model = model
        self.memory = memory
        self.rewardInvalid = rewardInvalid
        self.agentType = agentType
        gridSize = game.getGridSize()

        self.allActions = Action.makeAllActions(gridSize)
        self.actionSize = len(self.allActions)

        self.loss = 0
        self.recentLoss = 0
        
        


        self.dir = os.path.dirname(os.path.realpath(__file__))+"\\"

    def getLoss(self):
        tmp = self.loss
        self.recentLoss += tmp
        self.loss = 0
        return tmp

    def getRecentLoss(self):
        tmp = self.recentLoss
        self.recentLoss = 0
        return tmp


    def getType(self):
        return self.agentType

    def invalidMove(self, index, agentType, gameState):
        action = self.allActions[index]
        action = self.makeActionReadyForGame(agentType, action, gameState)

        return not self.game.isLegalMove(agentType, action)




    def move(self, agentType, currentStateVector, epsilon):



        if (not self.humanPlaying) and self.game.getRandom() and (random.random() <= epsilon):

            return self.getRandomAction(agentType, currentStateVector, self.getRandomMoveSize())


        else:
            if Action.greedy:
                q = self.model.predictOne(currentStateVector, self.sess)
                q = q.flatten()
                values = None

                if self.game.printQ:
                    self.game.printQ = False
                    print(agentType)
                    print(self.game.state)
                    print(currentStateVector)
                    values, indices = self.sess.run(tf.nn.top_k(q, len(q)-1))
                    values = values.tolist()
                    indices = indices.tolist()
                    for i in indices:
                        print(q[i], self.allActions[i])

                # try the first without sorting the q's
                greedyAction = np.argmax(q)

                if self.invalidMove(greedyAction, agentType, self.game.getState()):
                    # now we have to sort
                    # sort by best action, then run down the list until an action works
                    if values == None:
                        values, indices = self.sess.run(tf.nn.top_k(q, Action.TOP_K_SELECTION))
                        values = values.tolist()
                        indices = indices.tolist()

                    for i in range(Action.TOP_K_SELECTION-1):
                        self.memory.addSample((currentStateVector, greedyAction, self.rewardInvalid, None))
                        self.learn()
                        greedyAction = indices[i+1]
                        if not self.invalidMove(greedyAction, agentType, self.game.getState()):
                            return greedyAction

                    # no action was valid, so try the 8 basic moves to speed things up
                    return self.getRandomAction(agentType, currentStateVector, Action.PAWN_MOVES)
                else:
                    return greedyAction

            else:
                '''
                action = self.sample(values)

                while self.invalidMove(indices[action], agentType, self.game.getState()):
                    self.memory.addSample((currentStateVector, action, Action.INVALID_PENALTY, None))
                    del values[action]
                    del indices[action]

                    if len(values) == 0:
                        print("BUG!")
                        return -1, None

                    action = self.sample(values)
                return indices[action], self.allActions[indices[action]]
                '''




    def getRandomAction(self, agentType, currentStateVector, numActions):

        randomAction = random.randint(0, numActions)

        cnt = 0
        actionsTried = []

        while self.invalidMove(randomAction, agentType, self.game.getState()):
            cnt += 1
            self.memory.addSample((currentStateVector, randomAction, self.rewardInvalid, None))
            self.learn()
            actionsTried.append(randomAction)
            while True:
                randomAction = random.randint(0, self.getRandomMoveSize())
                if randomAction not in actionsTried:
                    break
                if cnt > 1000:
                    return -1

        return randomAction

    def getRandomMoveSize(self):
        if random.random() < Action.MOVE_PROBABILITY:
            moveSize = Action.PAWN_MOVES
        else:
            moveSize = self.actionSize-1
        return moveSize

    def sample(self, distribution):
        if sum(distribution) != 1:
            distribution = self.normalize(distribution)
        if float(sum(distribution)) == 0:
            return random.randint(0, len(distribution)-1)

        choice = random.random()
        i, total= 0, distribution[0]
        while choice >= total:
            i += 1
            total += distribution[i]
        return i


    def normalize(self, vector):
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]




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
                current_q[action] = reward + Action.GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q

        _, l = self.model.trainBatch(self.sess, x, y)
        self.loss += l



class TopAgent(Agent):
    def __init__(self, game, sess, model, memory, rewardInvalid, humanPlaying):
        Agent.__init__(self, game, BoardElement.AGENT_TOP, sess, model, memory, rewardInvalid)
        self.humanPlaying = humanPlaying
        self.goal = self.game.getGridSize() - 1



    def move(self, epsilon, state):
        actionIndex = Agent.move(self, BoardElement.AGENT_TOP, state, epsilon)
        if actionIndex == -1:
            return -1, None

        return actionIndex, self.allActions[actionIndex]

class BottomAgent(Agent):
    def __init__(self, game, sess, model, memory, rewardInvalid, humanPlaying):
        Agent.__init__(self, game, BoardElement.AGENT_BOT, sess, model, memory, rewardInvalid)
        self.humanPlaying = humanPlaying
        self.goal  = 0


    def move(self, epsilon, state):
        actionIndex = Agent.move(self, BoardElement.AGENT_BOT, state, epsilon)
        if actionIndex == -1:
            return -1, None

        return actionIndex, self.allActions[actionIndex]
