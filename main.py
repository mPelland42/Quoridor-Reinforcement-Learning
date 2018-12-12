

from Game import Qoridor
from Model import Model
from Memory import Memory
import tensorflow as tf

#import time

gridSize = 5
numWalls = 3 #number of walls each player starts with
gameSpeedSlow = 1 #time delay between moves when slowed down
humanPlaying = False
startWithDrawing = False #false to disable drawing of game on startup

restore = True #signifies if agent should be loaded from checkpoint


MEMORY = 500
NUM_EPOCHS = 2000 #how many games to play
GAMES_PER_EPOCH = 20
BATCH_SIZE = 150 #how many moves from memory to learn from
 #epsilon signifies chance that agent will make a random move
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0

LAMBDA = 0.00010 # decay of epsilon




game = Qoridor(gridSize, numWalls, gameSpeedSlow, startWithDrawing, humanPlaying)
model = Model(game.getStateSize(), gridSize, game.getActionSize(), BATCH_SIZE, restore, "agent")
mem = Memory(MEMORY)


#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    if not restore:
        sess.run(model._var_init)
    else:
        model.load(sess)
    
    game.setLearningParameters(sess, model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)

    cnt = 0
    print("Learning Initiated...")
    while cnt < NUM_EPOCHS:
        if cnt % GAMES_PER_EPOCH == 0 and cnt != 0 == 0:
            print('\nEpoch {} of {}'.format(cnt, NUM_EPOCHS))
            game.printDetails(GAMES_PER_EPOCH)
        game.run()
        cnt += 1
    #plt.plot(gr.reward_store)
    #plt.show()
    #plt.close("all")
    #plt.plot(gr.max_x_store)
    #plt.show()


