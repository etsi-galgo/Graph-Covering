# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:15:14 2024

@author: Alina Kasiuk
"""


import gym
import numpy as np
import igraph as ig
from envs import rendering

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
            self, graph, max_battery:int=1000, cover_reward=5, move_reward=-1, crash_reward=-100):
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
        self.max_battery = max_battery
        self.cover_reward = cover_reward
        self.move_reward = move_reward
        self.crash_reward = crash_reward
        
        # visualization
        self.viewer = None
        

    def reset(self):
   
        base_node = len(self.graph.vs)-1
        
        self.battery = self.max_battery
        self.state = self._get_state(base_node)
        self.graph.es["covered"] = False
        self.n_segments_covered = 0
        self.total_traveled_distance = 0
        self.coverage_distance = 0
        return base_node, self.state

    
    
    def step(self, action, node):

        
        action_space = self._get_actions(node)
        
#        err_msg = "%r (%s) invalid" % (action, type(action))
#        assert action_space.contains(action), err_msg   
        
        new_node = action_space[action]
        traveled_distance = self.graph.es.select(_within=[node,new_node])["weight"][0] 
        self.battery -= traveled_distance
        self.state = self._get_state(new_node)
        
        reward = self._get_reward(node, new_node)
        done =  bool(self.battery<0)
        if self.graph.es.select(_within=[node, new_node])["is_segment"][0] == True:
            self.n_segments_covered += 1
            self.coverage_distance += traveled_distance
            self.graph.es.select(_within=[node, new_node])["is_segment"] = False
            
        
        self.total_traveled_distance += traveled_distance
        self.graph.es.select(_within=[node, new_node])["covered"] = True
        
        return new_node, self.state, reward, done, {}
        
    
    def render(self, mode='rgb_array', show=True):

        screen_width = 640
        screen_height = 640  

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.trans = rendering.Transform()
            self.trans.set_translation(screen_width/2, screen_height/2)

            image = self._get_image()
            img = rendering.Image(image, screen_width, screen_height)
            img.add_attr(self.trans)
            self.viewer.add_geom(img)       
        
        if self.viewer is not None and show:
            image = self._get_image()
            img = rendering.Image(image, screen_width, screen_height)
            img.add_attr(self.trans)
            self.viewer.geoms[0] = img    
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None  
    
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

        charged = self._battery_level()

        return x, y, base, weight, is_segment, charged
    
    
    def _get_reward(self, node, new_node):  
        
        covered_segment = self.graph.es.select(_within=[node,new_node])["is_segment"][0]
        traveled_distance = self.graph.es.select(_within=[node,new_node])["weight"][0]
        overlapping = self.graph.es.select(_within=[node,new_node])["covered"][0]
        crash_free = self.battery>5
        
        r_cov = crash_free * traveled_distance * self.cover_reward * covered_segment
        r_move =  crash_free * traveled_distance * self.move_reward
        r_overlapping =  crash_free * traveled_distance * self.move_reward * overlapping
        
        r_crash = (not crash_free) * self.crash_reward
        
        
        return r_cov + r_move + r_crash + r_overlapping
    
    
    def _battery_level(self):       
        if self.battery<0:
            return -1
        if self.battery < self.max_battery/5:
            return 0
        if self.battery < 2*self.max_battery/5:
            return 1
        if self.battery < 3*self.max_battery/5:
            return 2
        if self.battery < 4*self.max_battery/5:
            return 3
        else:
            return 4
       
       
    def _get_image(self):
        image = "tmp.png"
        color_dict_vs = {True: "red", False: "black"}
        color_dict_es = {True: "red", False: "black"}
        edge_width = [2 + 7 * int(is_segment) for is_segment in self.graph.es["is_segment"]]
        vertex_color = [color_dict_vs[base] for base in self.graph.vs["base"]]
        edge_color = [color_dict_es[covered] for covered in self.graph.es["covered"]]
        ig.plot(
            self.graph,
            target=image,
            layout='auto',
            bbox=(1200, 1200),
            vertex_size = 20,
            vertex_frame_width=4.0,
            vertex_color = vertex_color,
            edge_width = edge_width,
            edge_color = edge_color,
            
        )

        return image
      
