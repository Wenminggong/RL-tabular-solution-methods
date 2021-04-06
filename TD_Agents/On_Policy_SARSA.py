#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:21:23 2020

@author: wenminggong

SARSA Agent
"""

import numpy as np
import copy


class SARSA():
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        
        
    def epsilon_policy(self, epsilon, Q, state):
        # sampling action
        A = np.ones(self.env.action_space.n, dtype = np.float32) * epsilon / self.env.action_space.n
        best_action_ide = np.argmax(Q[state])
        A[best_action_ide] += (1 - epsilon)
        p = float(np.random.rand(1))
        for a in range(self.env.action_space.n):
            if p <= A[a]:
                action = a
                break
            p -= A[a]
            
        return action
    
    
    def evaluate_q(self, Q, state, action, r, next_state, epsilon):
        next_action = self.epsilon_policy(epsilon, Q, next_state)
        Q[state, action] = Q[state, action] + self.alpha * (r + self.gamma * Q[next_state, next_action] - Q[state, action])
        return Q
        
        