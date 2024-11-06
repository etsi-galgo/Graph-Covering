# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:15:14 2024

@author: Alina Kasiuk
"""

import gym
import numpy as np
import igraph as ig
from envs import rendering
from utils import graph_v01

class GraphEnv(gym.Env):

    """
    GRAPH ENVIROMENT
    ----------
    Description:
    
    A drone with extended autonomy and limited battery starts on the base node
    and the goal is to find a strategy to opimize the coverage of parallel
    linear segments throught time.
    
    The problem is simplified by segment discretization and graph constuction 
    such an action of the RL agent correspond to an edge of the graph.
    ----------
    Graph:
    The graph is build with igraph library.
    
    Node features: 
        - 'x', 'y': continious = node coordinates
        - 'base': binary = is the node a base or not
        - 'here': binary = current position
        
    Edge features:
        - 'weight': continious = euqulidean distance between nodes
        - 'is_segment': binary = the edge is a segment to cover or not
        - 'covered': binary = the edge is alredy covered or not

    """
    #TODO: Change covered/not covered by covering X times: binary to integer

    def __init__(self, graph, qlearn_instance, max_battery:int=500, 
                 cover_reward=1, move_reward=-1, crash_reward=-1):
        
        self.__version__ = "0.0.1"
        """
        Parameters:
        ----------
        graph: prebuilded graph. Graph class from igraph  
        
        max_battery: maximal lenght can be covered in one tour
        
        cover_reward: a positive coefficient associated to segment covering
        move_reward:  a negative coefficient associated to every motion 
        crash_reward: big negative constant, reward when battery is over
        """
        self.qlearn_instance = qlearn_instance  # Store the QLearn instance
        self.graph = graph
        self.max_battery = max_battery
        self.cover_reward = cover_reward
        self.move_reward = move_reward
        self.crash_reward = crash_reward
        
        # visualization
        self.viewer = None
        self.visit_count = {}
        

    def reset(self):
        """
        Environment initialization / reset

        Returns initial state
        """
        
        base_node = len(self.graph.vs)-1 #Starting on the base station
        #TODO: Change this to find a base by features on any graph 
        #TODO: Choose between various bases
        
        self.graph.vs[base_node]['here'] = True #Position for monitoring (Used for visualization only)
        self.visit_count = {}
        self.battery = self.max_battery #Initialize with max battery level

        self.state = self._get_state(base_node) #Initial state
        
        self.graph.es["covered"] = 0 #All edges are not covered
        
        # Metrics initialization:
        self.n_segments_covered = 0 #Covered segments counter
        self.total_traveled_distance = 0 #Traveled distance counter
        self.coverage_distance = 0 #Covered distance counter
        self.n_segments = sum(self.graph.es["is_segment"]) #Number of segments to cover
        
        #TODO: Node is included to the state. Remove it from here
        return self.state

    
    
    def step(self, action, node):
        """
        Every step apply an action and get the reward
        
        Parameters
        ----------
        action : chosen action
        node : the current node on a graph

        Returns
        -------
        new_node : node after applying the given action
        reward : Reward received
        done : Episode Termination Flag
        
        """
        self.visit_count = self.qlearn_instance.get_visitation_count(''.join(map(str, self.state)) , action)
        action_space = self._get_actions(node) #Depends on number of edges of every node
        #TODO: avoid counting edges two times 
        
        new_node = action_space[action] #Change a node appling a given action
        self.state = self._get_state(new_node) #State change 
 #       print("State: ", self.state, "; Action: ", action)
 #       print("Visited: ", self.visit_count)
        
        #The weight of chosen edge is equal to traveled distance in this step:
        traveled_distance = self.graph.es.select(_within=[node,new_node])["weight"][0] 
        
        self.battery -= traveled_distance #Battery consumption

        reward = self._get_reward(node, new_node) #Compute the reward 
        
        #Change position for monitoring
        self.graph.vs[node]['here'] = False
        self.graph.vs[new_node]['here'] = True
               
        
        #Episode Termination:
        #If the battery is over
        #If all segments are covered
        if bool(self.battery<0) or sum(self.graph.es["is_segment"])==0:
            done = True
        else: done = False
        
        
        #If in this step we covered a segment:
        if self.graph.es.select(_within=[node, new_node])["is_segment"][0] == True:
            self.n_segments_covered += 1 #Covered segments counter
            self.coverage_distance += traveled_distance #Covered distance counter
            #The edge is not a segment anymore (not interesting to cover):
            self.graph.es.select(_within=[node, new_node])["is_segment"] = False 
        
        #Recharging on the base:
        if self.graph.vs[new_node]['base'] == True:
            self.battery = self.max_battery 
         #   print("recharge")
        
        
        self.total_traveled_distance += traveled_distance #Traveled distance counter
        if self.graph.es.select(_within=[node, new_node])["covered"][0]==0:
            self.graph.es.select(_within=[node, new_node])["covered"] = 1 #Edge is covered
        elif self.graph.es.select(_within=[node, new_node])["covered"][0]==1:
            self.graph.es.select(_within=[node, new_node])["covered"] = 2 #Edge is covered twice
        #TODO: Change covered/not covered by covering X times 
        
        return self.state, reward, done, {}
        
    
    def render(self, mode='rgb_array', show=True):
        """
        Visualization. See rendering.py to get the details
        Doesn't work on Linux

        Parameters:
        ----------
        mode : The default is 'rgb_array' to show a sequence of images in the 
                viewer window. Not change this
                
        show: showing or not the viewer. Set False to train faster
        """
        
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
        """
        Important to close the viewer
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None  
    
    
    def _get_actions(self, node): 
        """
        Parameters
        ----------
        node : the current node on a graph

        Returns
        -------
        action_dict : action distionary depending of the number
        of edges of every node

        """
        connected_nodes = np.asarray(np.where(np.asarray(self.graph.get_adjacency()[node]) == 1))
        connected_nodes = connected_nodes.tolist()[0] 
        
        action_dict = {}
        for index, element in enumerate(connected_nodes):
            action_dict[index] = element
            
        return action_dict
    
    
    def _get_state(self, node):
        """

        Parameters
        ----------
        node : the current node on a graph

        Returns
        -------
        A tuple state = [node, inciden edges features, battry level] 
        """
        
        # Edge features
        connected_nodes = np.asarray(np.where(np.asarray(self.graph.get_adjacency()[node]) == 1))
        connected_nodes = connected_nodes.tolist()[0]

        # Initialize edges features with zeroes 
        is_segment = np.zeros(len(connected_nodes))
        covered = np.zeros(len(connected_nodes))
        
        # For all edges inciden with a given node
        for i in range(len(connected_nodes)):
            #Is the edge an uncovered segment:
            is_segment[i] = self.graph.es.select(_within=[node,connected_nodes[i]])["is_segment"][0]
            #Is the edge already covered or not
            covered[i] = self.graph.es.select(_within=[node,connected_nodes[i]])["covered"][0]

        charged = self._battery_level()

        return node, is_segment, charged
    
    def _distance_to_the_base(self, node):
        (self.graph.vs[node]['x']**2+self.graph.vs[node]['y']**2)
        
        
    
    def _get_reward(self, node, new_node):
        """
        The normalized reward function
        
        Parameters
        ----------
        node : current node
        new_node : a node after applying a chosen action
        
        Returns the normalized reward value
        """
        current_edge = self.graph.es.select(_within=[node, new_node])
        covered_segment = current_edge["is_segment"][0]
        
        # Penalize for low battery
        discharged = self.battery < 100        
        shortest_path = graph_v01.shortest_path(self.graph, new_node)
        
        segments_left = sum(self.graph.es["is_segment"])
        
        crash_free = self.battery > shortest_path
        
        # Coverage reward normalized (between 0 and 1)
        r_cov = (self.n_segments - segments_left) * covered_segment / self.n_segments
        
        # Movement penalty 
        r_move = 0.5 + 0.5 *current_edge["covered"][0]
        
        # Battery penalty normalized (between 0 and -1)
        battery_penalty = shortest_path / (self.battery if self.battery > 0 else 1)
        
        # If everything is covered, return a large normalized reward
        if segments_left == 0:
            return 10  # Normalize max reward to 1
        
        # If battery is not enough to make it back to base, return a large penalty
        if not crash_free:
            return -5  # Normalize crash to -1
        
        # Add visitation bonus normalized (between 0 and 1)
        visitation_bonus = 1 / (1 + self.visit_count)
        
        # If battery is discharged, apply battery penalty
        if discharged:
            return -battery_penalty
        
        # Combine the normalized rewards and penalties, scaling the final result between -1 and 1
        total_reward = 0.8 * r_cov - 0.05 * r_move + 0.15 * visitation_bonus
        
        # Ensure total reward is between -1 and 1
        return max(min(total_reward, 1), -1)
        
        
    def _battery_level(self):   
        """
        Battery discretisation by levels
        """
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
        """
        Paint the graph
        ----------
        Returns rgb image
        """
        
        image = "tmp.png"
        color_dict_vs = {True: "green", False: "black"}
        color_dict_es = {2: "red", 1: "red4", 0:"black"}
        edge_width = [2 + 7 * int(is_segment) for is_segment in self.graph.es["is_segment"]]
        vertex_color = [color_dict_vs[here] for here in self.graph.vs["here"]]
        edge_color = [color_dict_es[covered] for covered in self.graph.es["covered"]]
        
        ig.plot(
            self.graph,
            target=image,
            layout='auto',
            bbox=(1200, 1200),
            vertex_size = 20,
            vertex_label = ["B"* int(base) for base in self.graph.vs["base"]],
            vertex_label_color = "red",
            vertex_frame_width=4.0,
            vertex_color = vertex_color,
            edge_width = edge_width,
            edge_color = edge_color,
            
        )
    
        return image
      
