# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina
"""

from graphs.graph import Graph

g = Graph()
g_delaunay = g.build_delaunay()
g.plot_the_graph(g_delaunay)