# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:15:14 2024

@author: Alina Kasiuk
"""


import gym
import numpy as np

class GraphEnv(gym.Env):

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
            self, graph, max_batery:int=100, cover_reward=5, move_reward=-1, crash_reward=-100):
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
        self.max_battery = max_batery
        self.cover_reward = cover_reward
        self.move_reward = move_reward
        self.crash_reward = crash_reward
        

    def reset(self):
        """
        Reset the environment. Use a random seed if you want to reproduce the random actions for 
        initializing the environment (such as random initial position of the agent).
        """
        # Vertex features
        
        base_node = len(self.graph.vs)-1
        
        self.battery = self.max_battery
        self.state = self._get_state(base_node)

        return base_node, self.state
    
    
    def step(self, action, node):
        
        action_space = self._get_actions(node)
        
#        err_msg = "%r (%s) invalid" % (action, type(action))
#        assert action_space.contains(action), err_msg   
        
        new_node = action_space[action]
        self.battery -= 1
        self.state = self._get_state(new_node)
        
        reward = self._get_reward(node, new_node)
        done =  bool(self.battery<0)
        
        #TODO: change covered segment into zero
        
        return new_node, self.state, reward, done, {}
        
    
    def _get_actions(self, node): 
        connected_nodes = np.asarray(np.where(np.asarray(self.graph.get_adjacency()[node]) == 1))
        connected_nodes = connected_nodes.tolist()[0] 
        
        action_dict = {}
        for index, element in enumerate(connected_nodes):
            action_dict[index] = element
            
        return action_dict
    
    
    def _get_state(self, node):
        x = self.graph.vs[node]['x']
        y = self.graph.vs[node]['y']
        base = self.graph.vs[node]['base']
        
        # Edge features
        connected_nodes = np.asarray(np.where(np.asarray(self.graph.get_adjacency()[node]) == 1))
        connected_nodes = connected_nodes.tolist()[0]
        weight = np.zeros(len(connected_nodes))
        is_segment = np.zeros(len(connected_nodes))
#        to_base = np.zeros(len(connected_nodes))
        for i in range(len(connected_nodes)):
            weight[i] = self.graph.es.select(_within=[node,connected_nodes[i]])["weight"][0]
            is_segment[i] = self.graph.es.select(_within=[node,connected_nodes[i]])["is_segment"][0]
#            to_base[i] = self.graph.es.select(_within=[n,connected_nodes[i]])["is_segment"][0]

        return x, y, base, weight, is_segment, self.battery
    
    
    def _get_reward(self, node, new_node):  
        
        covered_segment = self.graph.es.select(_within=[node,new_node])["is_segment"][0]
        traveled_distance = self.graph.es.select(_within=[node,new_node])["weight"][0]
        crash_free = self.battery>5
        
        r_cov = crash_free * traveled_distance * self.cover_reward * covered_segment
        r_move =  crash_free * traveled_distance * self.move_reward
        r_crash = (not crash_free) * self.crash_reward
        
        return r_cov + r_move + r_crash
        
        
        
        
        
    
