import pygame
import random
import time
from pygame.locals import *
pygame.init()
import sys
from Game import Qoridor

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)


game = Qoridor(screen)

currentAgent = 0

game.draw()

#print(game.getLegalMoves(0))

while True:
    currentAgent = (currentAgent + 1)%2
    screen.fill(0)
    game.draw()
    pygame.display.flip()
    pygame.display.update()
    time.sleep(1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()