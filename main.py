# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

from utils.graph import Graph
from envs.graph_env import GraphEnv
import random

g = Graph()
graph_path = g.build_delaunay()
g.plot_the_graph(graph_path)

env_ = GraphEnv(graph_path)

total_episodes = 5
env_iter = 20


for episode in range(total_episodes):
    node, state = env_.reset()
    print("Episode ", episode, " of ", total_episodes)
    
    for i in range(env_iter):
        env_.render()
        n_actions = graph_path.degree(node)
        # Pick a random action 
        action = random.choice(range(n_actions))        
        node, state, reward, done, _ = env_.step(action, node)
        print("action: ", action)
        print("node:", node)
        x, y, _, _, _, battery = state
        print("coordinates: (%r, %r)" % (x, y))
        print("battery level:", battery) 
        
env_.close()
