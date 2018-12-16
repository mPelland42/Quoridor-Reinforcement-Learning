

from Game import Qoridor
from Model import Model
from Memory import Memory
import tensorflow as tf

#import time

gridSize = 4
numWalls = 2 #number of walls each player starts with
gameSpeedSlow = 1 #time delay between moves when slowed down
humanPlaying = False
startWithDrawing = False #false to disable drawing of game on startup

restore = False #signifies if agent should be loaded from checkpoint


MEMORY = 1000
BATCH_SIZE = 300 #how many moves from memory to learn from

PREDICTION_MEMORY = 1000
PREDICTION_BATCH_SIZE = 500

NUM_EPOCHS = 10000 #how many games to play
GAMES_PER_EPOCH = 20
 #epsilon signifies chance that agent will make a random move
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0

LAMBDA = 0.0001 # decay of epsilon




game = Qoridor(gridSize, numWalls, gameSpeedSlow, startWithDrawing, humanPlaying)
model = Model(game.getStateSize(), (gridSize*2)-1, game.getActionSize(), BATCH_SIZE, restore, "agent")
mem = Memory(MEMORY, PREDICTION_MEMORY, PREDICTION_BATCH_SIZE)


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


