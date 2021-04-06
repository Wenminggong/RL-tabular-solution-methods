#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:10:32 2020
@author: wenminggong
dynamic programming with value iteraction
"""
import copy
import numpy as np
import pprint


class DPvalueiteration():
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        
    def onestepahead(self, V):
        candi_values = np.zeros(self.env.action_space.n)
        currentstate = copy.deepcopy(self.env._state)
        for action in range(self.env.action_space.n):
            next_state, reward, done, info = self.env.step(action)
            candi_values[action] = reward + self.gamma * V[next_state]
            self.env._state = copy.deepcopy(currentstate)
        return candi_values
        
    def valuefun(self, delta, V):
        "a sweep for the state spaces"
            
        for state in range(self.env.observation_space.n):
            self.env._state = state
            last_vs = copy.deepcopy(V[state])
            candi_values = self.onestepahead(V)
            V[state] = np.max(candi_values)
            delta = max(delta, abs(last_vs-V[state]))
            
        return V, delta
    
    def optimalpolicy(self, optimalV):
        optimalactions = []
        self.env.reset()
        while True:
            candi_values = self.onestepahead(optimalV)
            # print("candi values are:")
            # pprint.pprint(candi_values)
            optimalaction = np.argmax(candi_values)
            optimalactions.append(optimalaction)
            next_state, reward, done, info = self.env.step(optimalaction)
            if done:
                break
        
        print("optimal actions: {}".format(optimalactions))
        return optimalactions
        
