#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:03:18 2020

@author: wenminggong

find the optimal superparameters: alpha and gamma
"""

from lib.envs.maze import Maze2DEnv
import numpy as np
from matplotlib import pyplot as plt
from On_Policy_SARSA import SARSA
from Off_Policy_QLearning import QLearning
from Mazes import *
import pprint


def Q_learning_process(env, alpha, gamma):
    num_steps_qlearning = np.zeros(21)
    cum_rewards_qlearning = np.zeros(21)
    
    for i in range(10):
        num_steps_list = []
        cum_rewards_list = []
        # create QLearning agent
        Ql = QLearning(env, alpha, gamma)
        # training Q-Learning
        Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
        for episode in range(1001):
            epsilon = 0.1
            
            state = env.reset()
            cum_r = 0
            count_steps = 0
            for st in range(1000):
                action = Ql.epsilon_policy(epsilon, Q, state)
                next_state, r, done, info = env.step(action)
                Q = Ql.evaluate_q(Q, state, action, r, next_state)
                cum_r += r
                state = next_state
                count_steps = st + 1
                if done:
                    break
            if episode % 50 == 0:
                num_steps_list.append(count_steps)
                cum_rewards_list.append(cum_r)
        num_steps_qlearning += np.array(num_steps_list)
        cum_rewards_qlearning += np.array(cum_rewards_list)
    
    return num_steps_qlearning / 10, cum_rewards_qlearning / 10


def main():
    import sys
    print(sys.version)
    
    # get simple maze
    # simple_maze, s, g = Simple_maze()
    
    # # get complex maze
    complex_maze, s, g = Complex_maze()
    
    # set maze env
    env = Maze2DEnv(height = 22, width = 22, start=s, goal=g)
    env.set_env(complex_maze)
    
    # hyper-parameter
    alpha = 0.1
    # "show the learning process"
    t = np.linspace(0, 1000, 21)
    plt.figure()
    
    # alpha = 0.01
    gamma = 0.3
    num_steps_qlearning, cum_rewards_qlearning = Q_learning_process(env, alpha, gamma)
    # plt.plot(t, num_steps_qlearning, c = 'm', marker = 's', label="alpha =" + str(alpha))
    plt.plot(t, num_steps_qlearning, c = 'm', marker = 's', label="gamma =" + str(gamma))
    
    # alpha = 0.05
    gamma = 0.5
    num_steps_qlearning, cum_rewards_qlearning = Q_learning_process(env, alpha, gamma)
    # plt.plot(t, num_steps_qlearning, c = 'g', marker = '*', label="alpha =" + str(alpha))
    plt.plot(t, num_steps_qlearning, c = 'g', marker = '*', label="gamma =" + str(gamma))
    
    # alpha = 0.2
    gamma = 0.7
    num_steps_qlearning, cum_rewards_qlearning = Q_learning_process(env, alpha, gamma)
    # plt.plot(t, num_steps_qlearning, c = 'r', marker = 'o', label="alpha =" + str(alpha))
    plt.plot(t, num_steps_qlearning, c = 'r', marker = 'o', label="gamma =" + str(gamma))
    
    # alpha = 0.5
    gamma = 0.9
    num_steps_qlearning, cum_rewards_qlearning = Q_learning_process(env, alpha, gamma)
    # plt.plot(t, num_steps_qlearning, c = 'b', marker = '+', label="alpha =" + str(alpha))
    plt.plot(t, num_steps_qlearning, c = 'b', marker = '+', label="gamma =" + str(gamma))
    
    # alpha = 0.9
    gamma = 0.99
    num_steps_qlearning, cum_rewards_qlearning = Q_learning_process(env, alpha, gamma)
    # plt.plot(t, num_steps_qlearning, c = 'y', marker = 'x', label="alpha =" + str(alpha))
    plt.plot(t, num_steps_qlearning, c = 'y', marker = 'x', label="gamma =" + str(gamma))
    
    plt.legend()
    plt.ylabel("Steps to the goal")
    plt.xlabel("Learning eposides")
    plt.title("Q-Learning with alpha = 0.1 in complex maze")
    plt.savefig('Q_gamma_complex_maze', format='eps')
    plt.show()
    
if __name__ == "__main__":
    main()


