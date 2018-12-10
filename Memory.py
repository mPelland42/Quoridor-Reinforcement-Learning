# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:02:18 2018

@author: Sam
"""
import random

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def addSample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            del self._samples[0]

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
        
    def getTotalMem(self):
        return len(self._samples)