# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina
"""

from utils.graph import Graph
from envs.graph_env import GraphEnv

g = Graph()
g_delaunay = g.build_delaunay()
g.plot_the_graph(g_delaunay)

env_ = GraphEnv(g_delaunay)
node, state = env_.reset()

