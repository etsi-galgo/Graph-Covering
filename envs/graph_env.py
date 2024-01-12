# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:15:14 2024

@author: Alina Kasiuk
"""


import gym
import numpy as np

class DroneEnv(gym.Env):

    """
    GRAPH ENVIROMENT
    
    Description:

       
    Reward:
    ***    

    Starting State: on the base node
    ***


    Episode Termination:
    ***
    """

    def __init__(
            self, graph, max_batery:int=100, cover_reward=5, move_reward=-1, crash_reward=-100, action_prob:int=1
        ):
        self.__version__ = "0.0.1"
        """
        cover_reward: the constant value multiplied by the cell of the relevance map covered. This reward is only used 
            the first time the drone visit the cell.
        move_reward: the reward of moving the drone. Should be negative to enforce short paths.
        crash_reward: the reward for crashing, i.e., when the agent hit an obstacle, the ground (out of battery), or map borders.
        action_prob: the probability that the drone performs the action that selects. If lower than one, serves as a random simulator 
            of things that can go wrong.
        """
        self.graph = graph
        self.max_batery = max_batery
        

    def reset(self):
        """
        Reset the environment. Use a random seed if you want to reproduce the random actions for 
        initializing the environment (such as random initial position of the agent).
        """
        # Vertex features
        
        x = self.graph.vs[0]['x']
        y = self.graph.vs[0]['y']
        base = True
        
        # Edge features
        connected_nodes = np.asarray(np.where(np.asarray(self.graph.get_adjacency()[0]) == 1))
        connected_nodes = connected_nodes.tolist()[0]
        weight = np.zeros(len(connected_nodes))
        is_segment = np.zeros(len(connected_nodes))
        to_base = np.zeros(len(connected_nodes))
        for i in range(len(connected_nodes)):
            weight[i] = self.graph.es.select(_within=[0,connected_nodes[i]])["weight"][0]
            is_segment[i] = self.graph.es.select(_within=[0,connected_nodes[i]])["is_segment"][0]
            to_base[i] = self.graph.es.select(_within=[0,connected_nodes[i]])["is_segment"][0]
            
        

        self.state = x, y, base, weight, is_segment, to_base, self.max_batery

        return self.state


