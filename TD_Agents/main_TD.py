#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:23:34 2020

@author: wenminggong

main function for TD methods
"""

from lib.envs.maze import Maze2DEnv
import numpy as np
from matplotlib import pyplot as plt
from On_Policy_SARSA import SARSA
from Off_Policy_QLearning import QLearning


def main():
    import sys
    print(sys.version)
    # hyper-parameter
    gamma = 0.99
    alpha = 0.05
    # create maze environment
    env = Maze2DEnv()
    env.set_env()
    
    num_steps_sarsa = []
    cum_rewards_sarsa = []
    num_steps_qlearning = []
    cum_rewards_qlearning = []
    
    # create SARSA agent
    Sarsa = SARSA(env, alpha, gamma)
    # initialize Q function
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    # training SARSA
    for episode in range(2000):
        # if episode < 1200:
        #     epsilon = 0.1
        # elif (episode >= 1200) and (episode < 1800):
        #     epsilon = 0.05
        # else:
        #     epsilon = 0
        epsilon = 0.05
            
        state = env.reset()
        cum_r = 0
        for st in range(5000):
            action = Sarsa.epsilon_policy(epsilon, Q, state)
            next_state, r, done, info = env.step(action)
            Q = Sarsa.evaluate_q(Q, state, action, r, next_state, epsilon)
            cum_r += r
            state = next_state
            if done:
                count_steps = st + 1
                break
        
        num_steps_sarsa.append(min(5000, count_steps))
        cum_rewards_sarsa.append(cum_r)
    
    # create QLearning agent
    Ql = QLearning(env, alpha, gamma)
    # training Q-Learning
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    for episode in range(2000):
        epsilon = 0.05
            
        state = env.reset()
        cum_r = 0
        for st in range(5000):
            action = Ql.epsilon_policy(epsilon, Q, state)
            next_state, r, done, info = env.step(action)
            Q = Ql.evaluate_q(Q, state, action, r, next_state)
            cum_r += r
            state = next_state
            if done:
                count_steps = st + 1
                break
        num_steps_qlearning.append(min(5000,count_steps))
        cum_rewards_qlearning.append(cum_r)
    
    # "show the learning process"
    t = np.linspace(0, 1999, 2000)
    plt.figure()
    # plt.plot(t, num_steps_sarsa, c = 'r', label="SARSA")
    # plt.plot(t, num_steps_qlearning, c = 'g', label="QLearning")
    plt.plot(t, cum_rewards_sarsa, c = 'r', label="SARSA")
    plt.plot(t, cum_rewards_qlearning, c = 'g', label="QLearning")
    plt.legend()
    plt.ylabel("steps to the goal")
    plt.xlabel("learning eposides")
    plt.title("TD Learning Process")
    plt.show()


if __name__ == "__main__":
    main()