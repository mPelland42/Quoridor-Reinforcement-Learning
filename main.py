import pygame
from pygame.locals import *
pygame.init()
import sys
from Game import Qoridor

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

game = Qoridor(screen)

game.draw()

print(game.getLegalMoves(0))

while True:
    screen.fill(0)
    game.draw()
    pygame.display.flip()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()