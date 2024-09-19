# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:15:49 2024

@author: Alina
"""

import numpy as np
import random
import math
import igraph as ig
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

class MyGraph:
    def __init__(self, vertices=None, base=[0,0], width=0, height=0, n_lines=0, max_segs_per_line=0):
        """
        Initialize the MyGraph instance with a map of the segments located on parrallel lines.
        
        Parameters:
        - given_graph: An igraph.Graph instance to initialize from. 
        Contanes segments located on parrallel lines with verces coordinates and the base station.
        Supposed to be a graph saved from the previos experiment
        
        If None, a new graph is created to start a new experiment. 
        In this case the following parameters are requiered:
        - width: Width of the area to create vertices.
        - height: Height of the area to create vertices.
        - n_lines: Number of parallel lines.
        - max_segs_per_line: Maximum segments per line.
        - base: Coordinates of the base station. # One base at the moment. Make a tuple.
        """
        
        self.vertices = np.empty((0, 2))
        self.segments = np.empty((0, 2))
        self.segment_graph = ig.Graph(directed=False)
        
        if vertices is None:
            self.vertices_on_lines(width, height, n_lines, max_segs_per_line)
            self.segments_on_parallel_lines_graph()
            self.add_base(base)
        else:
            self.vertices = vertices
            self.segments_on_parallel_lines_graph()
            

    def vertices_on_lines(self, width, height, n_lines, max_segs_per_line):
        """
        Generate vertices on parallel lines within a given area.
        
        Parameters:
        - width: Width of the area.
        - height: Height of the area.
        - n_lines: Number of parallel lines.
        - max_segs_per_line: Maximum segments per line.
        """
        seg_heights = sorted(random.sample(range(1,height), n_lines)) # hight of each line randomly
        
        for i in range(n_lines):
            # distribute segments randomly
            new_line = np.array(sorted(random.sample(range(width - 1), max_segs_per_line * 2))) 
            for j in range(new_line.shape[0]):
                # array of vertices: [[x1,y1], [x2,y2], ..., [xn,yn]]
                self.vertices = np.append(self.vertices, [[new_line[j], seg_heights[i]]], axis=0)



    def segment_edges(self):
        """
        Create segment edges based on the vertices.
        """
        self.n_segments = int(self.vertices.shape[0] / 2)
        self.segments = np.zeros([self.n_segments, 2], dtype=int)
        for i in range(self.n_segments):
            #array of edges that represent segments: [[v1,v2], [v3,v4], ...]
            self.segments[i] = [int(i * 2), int(i * 2 + 1)]

    def segments_on_parallel_lines_graph(self):
        """
        Create a graph with vertices on parallel lines and add it to the segment_graph.
        
        Parameters:
        - width: Width of the area.
        - height: Height of the area.
        - n_lines: Number of parallel lines.
        - max_segs_per_line: Maximum segments per line.
        """
       
        self.segment_edges()
        self.segment_graph.add_vertices(self.vertices.shape[0])
        self.segment_graph.add_edges(self.segments.tolist())
        
        # Set x and y coordinates for the vertices
        self.segment_graph.vs['x'] = self.vertices[:, 0]
        self.segment_graph.vs['y'] = self.vertices[:, 1]
        self.segment_graph.vs['base'] = False
        self.segment_graph.es["is_segment"] = True
        self.segment_graph.vs['here'] = False
        
        self.weigh_edges(self.segment_graph)
        print(self.segment_graph.es["is_segment"])
        print(self.segment_graph.es["weight"])

    def add_base(self, base):
        """
        Add a base station vertex to the graph.
        
        Parameters:
        - base: Coordinates of the base station.
        """
        n = len(self.segment_graph.vs)
        self.segment_graph.add_vertices(1)
        self.segment_graph.vs[n]['x'] = base[1]
        self.segment_graph.vs[n]['y'] = base[0]
        self.segment_graph.vs[n]["base"] = True
        self.segment_graph.vs[n]['here'] = False

    def weigh_edges(self, graph=None):
        """
        Define the "weight" attribute as the Euclidean distance between the connected vertices.
        
        Parameters:
        - graph: igraph.Graph graph to weigh. If None, segment graph (without roads) is used.
        """
        if graph is None:
            graph = self.segment_graph

        for i, edge_nodes in enumerate(graph.get_edgelist()):
            x_dist = graph.vs[edge_nodes[1]]['x'] - graph.vs[edge_nodes[0]]['x']
            y_dist = graph.vs[edge_nodes[1]]['y'] - graph.vs[edge_nodes[0]]['y']
            distance = math.sqrt(x_dist ** 2 + y_dist ** 2)
            graph.es[i]["weight"] = distance
            
        return graph.es["weight"]

    def delaunay_graph(self):
        """
        Creates a Delaunay triangulation and marks edges that exist in the original graph
        with "is_segment" attribute.
        
        
        Returns:
        - Connected graph. Consist of segment graph and roads (connections between ssegments)
        The roads are abtaint using Delaunay triangulation.
        """
        
        self.g_delaunay = self.segment_graph.copy()
        layout = self.g_delaunay.layout_auto()
        self.delaunay = Delaunay(layout.coords)

        # Add edges for each triangle in the Delaunay triangulation
        for tri in self.delaunay.simplices:
            self.g_delaunay.add_edges([
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[0], tri[2]),
            ])

        # Simplify the graph (remove duplicate edges)
        self.g_delaunay.simplify()

        # Initialize the edge attribute
        self.g_delaunay.es["is_segment"] = [False] * self.g_delaunay.ecount()

        # Create a set of edges from the original graph
        original_edges = set(tuple(sorted(edge)) for edge in self.segment_graph.get_edgelist())

        # Update the 'is_segment' attribute for edges that are in the original graph
        for idx, edge in enumerate(self.g_delaunay.get_edgelist()):
            if tuple(sorted(edge)) in original_edges:
                self.g_delaunay.es[idx]["is_segment"] = True
        
        self.g_delaunay.es['covered'] = 0
        # Weigh edges in the Delaunay graph
        self.weigh_edges(self.g_delaunay)

        return self.g_delaunay

    def all_vertices_have_attribute(self, graph, attribute_name):
        """
        Check if all vertices in the graph have a given attribute.
        """
        return all(attribute_name in v.attributes() for v in graph.vs)

    def all_edges_have_attribute(self, graph, attribute_name):
        """
        Check if all edges in the graph have a given attribute.
        """
        return all(attribute_name in e.attributes() for e in graph.es)

def plot_graph(graph):
    """
    Plot the graph with matplotlib.
    """
    color_dict_vs = {True: "red", False: "black"}

    if 'base' in graph.vs.attributes():
        vertex_color = [color_dict_vs[base] for base in graph.vs["base"]]
        vertex_label = ["\n     B" * int(base) for base in graph.vs["base"]]
    else:
        vertex_color = "black"
        vertex_label = ""

    if 'is_segment' in graph.es.attributes():
        edge_width = [2 + 10 * int(is_segment) for is_segment in graph.es["is_segment"]]
    else:
        edge_width = 2

    fig, ax = plt.subplots(figsize=(50, 50))  # Adjust size for practical visualization

    ig.plot(
        graph,
        target=ax,
        layout='auto',
        vertex_size=0.5,
        vertex_label=vertex_label,
        vertex_label_size=8,  # Adjusted size for better readability
        vertex_frame_width=1.0,  # Adjusted frame width for aesthetics
        vertex_color=vertex_color,
        edge_width=edge_width,
    )
    plt.show()
    
def shortest_path(graph, node):    
    """    
    Calculate the shortest path from a given node to the base 
    """  
    if not graph.is_connected():
        raise ValueError("The graph is not connected. Cannot calculate the shortest path to the base.")

    base_idx = graph.vcount()-1

    shortest_path = graph.get_shortest_paths(node, to=base_idx, weights=graph.es['weight'], output='vpath')

    # Print the shortest path as a list of vertices
    print("Shortest path in the undirected weighted graph:", shortest_path[0])

    # Print the total weight of the shortest path
    path_edges = graph.get_shortest_paths(node, to=base_idx, weights=graph.es['weight'], output='epath')[0]
    total_weight = sum(graph.es[edge]['weight'] for edge in path_edges)
    print("Total weight of the shortest path:", total_weight)
    
def total_length(graph):
    """    
    Calculate the total length of all edges in the graph
    """      
    if not 'weight' in graph.es.attributes():
        raise ValueError("The graph is not weighted. Cannot calculate the total length.")   
    return sum(graph.es["weight"])    
    
    
    
    
    