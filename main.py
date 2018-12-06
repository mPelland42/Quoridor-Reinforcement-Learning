

from Game import Qoridor
from Model import Model
from Memory import Memory

import tensorflow as tf



gridSize = 9
gameSpeed = 0
displayGame = False
humanPlaying = False


MEMORY = 50000
NUM_EPISODES = 300
BATCH_SIZE = 100
 #max decay, min decay
MAX_EPSILON = 1.0
MIN_EPSILON = 0.0
LAMBDA = 0.00001 # decay



        
game = Qoridor(gridSize, gameSpeed, displayGame, humanPlaying)
model = Model(game.getStateSize(), game.getActionSize(), BATCH_SIZE)
mem = Memory(MEMORY)


with tf.Session() as sess:
    sess.run(model._var_init)
    
    if not humanPlaying:
        game.setLearningParameters(sess, model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
    
        cnt = 0
        while cnt < NUM_EPISODES:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, NUM_EPISODES))
                game.printEpsilon()
            game.run()
            cnt += 1
        #plt.plot(gr.reward_store)
        #plt.show()
        #plt.close("all")
        #plt.plot(gr.max_x_store)
        #plt.show()
    else:
        print("testing, 1..2..3..")





