#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 00:11:04 2020
@author: wenminggong
main
"""
from lib.envs.maze import Maze2DEnv
from value_iteration import DPvalueiteration
import numpy as np
import pprint
from matplotlib import pyplot as plt


def main():
    import sys
    print(sys.version)
    
    maze = np.zeros((6, 6), dtype=np.bool)
    maze[0] = [1] * 6
    maze[1] = [1,0,0,0,0,1]
    maze[2] = [1,1,0,0,0,1]
    maze[3] = [1,0,1,0,0,1]
    maze[4] = [1,0,0,0,0,1]
    maze[5] = [1] * 6
    
    env = Maze2DEnv(height = 6, width = 6, start=(1,1), goal=(4,4))
    env.set_env(maze)
    # env = Maze2DEnv()
    # env.set_env()
    dpvalueiteration = DPvalueiteration(env)
    
    # figure_x = []
    # figure_y = []
    
    theta = 0.0001
    delta = 0
    V = np.zeros(env.observation_space.n, dtype=np.float32)
    count = 0
    while True:
        V, delta = dpvalueiteration.valuefun(delta, V)
        # if (count % 50 == 0):
        #     figure_x.append(count)
        #     optimalactions = dpvalueiteration.optimalpolicy(V)
        #     figure_y.append(len(optimalactions))
        
        if delta < theta:
            break
        count += 1
        if count > 100:
            print("iterate {} sweaps".format(count))
            break
        
    optimalactions = dpvalueiteration.optimalpolicy(V)
    print("the optimal value functions are:")
    pprint.pprint(V)
    print("optimal solutiion is {} steps".format(len(optimalactions)))
    pprint.pprint(optimalactions)
    
    # "show the learning process"
    # plt.figure()
    # plt.plot(figure_x, figure_y, c = 'r', maker='x', label="DP_value_iteration")
    # plt.legend()
    # plt.ylabel("steps to the goal")
    # plt.xlabel("learning eposides")
    # plt.title("DP_value_iteration Learning Process")
    # plt.show()
    
    
    

if __name__ == "__main__":
    main()