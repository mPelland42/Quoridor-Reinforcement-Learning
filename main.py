import pygame
import random
import time
from pygame.locals import *
pygame.init()
import sys

from Game import Qoridor

from Agents import TopAgent
from Agents import BottomAgent

import tensorflow as tf

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)


game = Qoridor(screen)

currentAgent = 0

game.draw()

#print(game.getLegalMoves(0))
print("Setting up agent networks...")
topAgent = TopAgent(game)
bottomAgent = BottomAgent(game)
print("completed")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    

    agents = [topAgent, bottomAgent]
    currentAgent = 0
    
    while True:
        screen.fill(0)
        game.draw()
        pygame.display.flip()
        pygame.display.update()
        
        # current agent goes
        move = agents[currentAgent].move(sess)
        
        # perform the action and update all agents according to this move
        game.performAction(currentAgent, move)
        for agent in agents:
            if agent != agents[currentAgent]:
                agent.updateFromEnemyMove(move)
            
            
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sess.close()
                pygame.quit()
                sys.exit()
        
        currentAgent = (currentAgent + 1)%2