# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:02:18 2018

@author: Sam
"""
import random
from GameState import BoardElement

class Memory:
    def __init__(self, max_memory, predictionMemory, predictionBatchSize):
        self._max_memory = max_memory
        self.predictionMemory = predictionMemory
        self.predictionBatchSize = predictionBatchSize
        self._samples = []
        
        self.stateWinMemory = list()

    def addSample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            del self._samples[0]

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            # include the most recent 3 for faster learning
            #return self._samples[-3:] + random.sample(self._samples, no_samples)
            return random.sample(self._samples, no_samples)
        
    def getTotalMem(self):
        return len(self._samples)
    
    def addStateWinMemory(self, stateWinMemory):
        for i in stateWinMemory[BoardElement.AGENT_TOP]:
            self.stateWinMemory.append(i)
            
        for i in stateWinMemory[BoardElement.AGENT_BOT]:
            self.stateWinMemory.append(i)
            
        if len(self.stateWinMemory) > self.predictionMemory:
            memDif = len(self.stateWinMemory) - self.predictionMemory
            del self.stateWinMemory[0:memDif]
            
        
    def sampleStateWinBatch(self):
        if self.predictionBatchSize > len(self.stateWinMemory):
            return random.sample(self.stateWinMemory, len(self.stateWinMemory))
        else:
            return random.sample(self.stateWinMemory, self.predictionBatchSize)