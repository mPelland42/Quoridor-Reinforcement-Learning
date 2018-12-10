

from Game import Qoridor
from Model import Model
from Memory import Memory

import tensorflow as tf

#import time

gridSize = 5
numWalls = 3
gameSpeedSlow = 1
humanPlaying = False
startWithDrawing = False


MEMORY = 300
NUM_EPOCHS = 1000
GAMES_PER_EPOCH = 20
BATCH_SIZE = 50
 #max decay, min decay
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0

LAMBDA = 0.000035 # decay




game = Qoridor(gridSize, numWalls, gameSpeedSlow, startWithDrawing, humanPlaying)
model = Model(game.getStateSize(), 2 * gridSize - 1, game.getActionSize(), BATCH_SIZE)
mem = Memory(MEMORY)


#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(model._var_init)
    
    if not humanPlaying:
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
    else:
        print("testing, 1..2..3..")


