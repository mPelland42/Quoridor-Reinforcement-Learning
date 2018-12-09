

from Game import Qoridor
from Model import Model
from Memory import Memory

import tensorflow as tf

#import time

gridSize = 9
gameSpeedSlow = 2
humanPlaying = False
startWithDrawing = False


MEMORY = 200000
NUM_EPISODES = 1000
BATCH_SIZE = 100
 #max decay, min decay
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0
LAMBDA = 0.0001 # decay




game = Qoridor(gridSize, gameSpeedSlow, startWithDrawing, humanPlaying)
model = Model(game.getStateSize(), game.getActionSize(), BATCH_SIZE)
mem = Memory(MEMORY)


#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(model._var_init)
    
    if not humanPlaying:
        game.setLearningParameters(sess, model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
    
        cnt = 0
        while cnt < NUM_EPISODES:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, NUM_EPISODES))
                game.printDetails()
            game.run()
            cnt += 1
        #plt.plot(gr.reward_store)
        #plt.show()
        #plt.close("all")
        #plt.plot(gr.max_x_store)
        #plt.show()
    else:
        print("testing, 1..2..3..")


