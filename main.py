import pygame
import random
import time
from pygame.locals import *
pygame.init()
import sys

from Game import Qoridor
import cProfile


#get user input on what kind of game to play


"""
gameType = input("Would you like to learn or play? ")
if gameType == "learn":
    numGames = input("how many games? ")
    storLoc = input("where should I store the results after playing? ")
else:
    readLoc = "what File should I read from? (type NONE to play dumb AI)"
"""










currentAgent = 0
AI = 0
HUMAN = 1
agents = [AI, AI]

from Agents import TopAgent
from Agents import BottomAgent

import tensorflow as tf

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)

scores = [0, 0]
game = Qoridor(screen)

# right now, a wall is draw to the screen at the beginning.. not sure how to fix
game.draw(0, (0, 0))


pygame.display.flip()
pygame.display.update()


'''
def getRandomAction(agent):
    wallMoves = [('w', (x, y), z) for x in range(7) for y in range(7) for z in [1, 2]]
    pawnMoves = [('p', x) for x in game.getPawnMoves(game.agents[agent])]
    moves = pawnMoves + wallMoves
    while 1:
        move = random.choice(moves)
        if game.isLegalMove(agent, move):
            return move
'''



print("Setting up agent networks...")
topAgent = TopAgent(game)
bottomAgent = BottomAgent(game)
print("completed\n")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


    super_agents = [topAgent, bottomAgent]
    currentAgent = 0

    while True:
        drawn = False
        if agents[currentAgent] == AI:
            
            agent = super_agents[currentAgent]
            action = agent.move(sess)
            game.performAction(currentAgent, action)
            print ("============")
        
            
            
            
            #if random.random() < .5:
            #    game.performAction(currentAgent, random.choice([('p', x) for x in game.getPawnMoves(game.agents[currentAgent])]))
            #else:
            #    action = getRandomAction(currentAgent)
            #    game.performAction(currentAgent, action)
            
            if game.endGame() != -1:
                scores[currentAgent] += 1
                print(scores)
                currentAgent = 0
                game = Qoridor(screen)
            else:
                currentAgent = (currentAgent + 1) % 2
            
            screen.fill(0)
            game.draw(currentAgent, pygame.mouse.get_pos())
            drawn = True

            time.sleep(1) # so we can see wtf is going on
            
        if game.maybeMoveChanged(currentAgent, pygame.mouse.get_pos()):
            screen.fill(0)
            game.draw(currentAgent, pygame.mouse.get_pos())
            drawn = True

        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and agents[currentAgent] == HUMAN:
                if game.playerAction(currentAgent, pygame.mouse.get_pos()):
                    if game.endGame() != -1:
                        scores[currentAgent] += 1
                        print(scores)
                        currentAgent = 0
                        game = Qoridor(screen)
                    else:
                        currentAgent = (currentAgent+1)%2
                    screen.fill(0)
                    game.draw(currentAgent, pygame.mouse.get_pos())
                    drawn = True

        if drawn:
            pygame.display.flip()
            pygame.display.update()
