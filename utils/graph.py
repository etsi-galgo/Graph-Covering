# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:42:29 2024

@author: Alina Kasiuk
"""

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
from scipy.spatial import Delaunay

class Graph():
    
    def __init__(self, lines = None, base=[0,25], l = 50, min_seg = 2, max_seg = 5, n = 5 , max_hight = 50):
        
    
        self.__version__ = "0.0.1"
       
        """
        l: max line lenght 
        min_seg: min segments per line
        max_seg: max segments per line
        n: number of lines
        max_hight: max hight of the higthest segment
        base: base station coordinates
        """
    
        self.base = base
        
        # segments generation
        if lines == None:
            self.lines = self._multiple_lines(l, min_seg, max_seg, n, max_hight)
        else:
            self.lines = lines
        
        self.graph = self._segment_to_graph()
        self.graph = self._add_base(self.graph, base) # Change this because in some cases we connct and than add base and connext it to all
        self.connected_graph = self.graph
    

    def _line_generate(self, lenght, m):
        """
        Creating a line with m random segments
        """
        ab = np.array(sorted(random.sample(range(0, lenght-1), m*2))) # random start and end of the segments
        ab = ab.reshape((m,2)) # start and end of the segments coordinates
        return ab

    def _multiple_lines(self, line_lenght, min_seg, max_seg, line_num, max_hight):
        """
        Creating line_num parallel lines with segments
        """        
        seg_hights = sorted(random.sample(range(0, max_hight), line_num))
        lines = pd.DataFrame(columns = ['Start','End', 'Height', 'Line'])
        for i in range(line_num):
            line =  pd.DataFrame(self._line_generate(line_lenght, random.randrange(min_seg, max_seg)), columns = ['Start','End'])
            line["Height"] = seg_hights[i]
            line["Line"] = i
            lines = pd.concat([lines,line], ignore_index=True)
        return lines

    def _graph_params(self):
        """
        Converting generated lines into a graph of ighaph
        """          
        n_vertices = self.lines.shape[0]*2
        starts = self.lines['Start'].to_numpy()
        ends = self.lines['End'].to_numpy()
        hight = self.lines['Height'].to_numpy()
        x = np.zeros(self.lines.shape[0]*2)
        y =  np.zeros(self.lines.shape[0]*2)
        
        for i in range(self.lines.shape[0]):
            x[i*2] = starts[i]
            x[i*2+1] = ends[i]
            y[i*2] = hight[i]
            y[i*2+1] = hight[i]   
                 
        edge = np.zeros([self.lines.shape[0],2])    
        for i in range(self.lines.shape[0]):
            edge[i] = [i*2,i*2+1]
        
        return n_vertices, edge, x, y
        
            
    
    def _segment_to_graph(self):      
        self.n_vertices, self.edge, x, y = self._graph_params()
        g = ig.Graph(self.n_vertices, self.edge)
        g.vs['x'] = x
        g.vs['y'] = y
        g.vs['base'] = False
        g.es['is_segment'] = True
        
        
        return g
    
    def _add_base(self, g, base):
        """
        Adding a base station vertex to a graph
        """            
        n=len(g.vs)
        g.add_vertices(1)
        g.vs[n]['x'] = base[1]
        g.vs[n]['y'] = base[0]
        g.vs[n]["base"] = True
        g.vs[0:n]["base"] = False
        return g

    def _weigh_edges(self, g):
        """
        Defining the "weight" attribute as the distance between nodes
        """         
        for i in range(len(g.get_edgelist())):
            edge_nodes = g.get_edgelist()[i]
            x_dist = g.vs[edge_nodes[1]]['x'] - g.vs[edge_nodes[0]]['x']
            y_dist = g.vs[edge_nodes[1]]['y'] - g.vs[edge_nodes[0]]['y']
            
            distance = math.sqrt(x_dist**2+y_dist**2)        
            g.es[i]["weight"] = distance

    def build_full(self):
        """
        Full connected graph generation
        """           
        g_comp = self.graphg.complementer(loops=False)
        g_full = self.graphg|g_comp 
        g_full.simplify()
        g_full = self.graph + g_full
        g_full.es["is_segment"] = [False if is_segment is None else is_segment for is_segment in g_full.es["is_segment"]]
        
        self._weigh_edges(g_full)
        return g_full

    def _delete_not_closest(self, g, n):
        """
        Deleting too long edges of full connected graph to satisfy n-closest rule
        """  
        for i in reversed(range(g.vcount())):
            v = g.vs[i]
            if v.degree()>n: 
                edge = v.incident()
                w = np.empty(0)
                for j in range(len(edge)):
                    w = np.append(w, edge[j]["weight"]) 
                sorted_weights = np.sort(w)
            
                max_lenght = sorted_weights[n-1]
            
                for k in reversed([ed.index for ed in edge]):
                    if (g.es[k]["weight"] > max_lenght) and (g.es[k]["is_segment"]==False):
                        g.delete_edges(k)


    def _add_segments(self, g, segments):
        for i in range(segments.shape[0]):
            g.es[g.get_eid(int(segments[i,0]), int(segments[i,1]))]["is_segment"] = True    
        g.es["is_segment"] = [False if is_segment is None else is_segment for is_segment in g.es["is_segment"]]
    
    def build_n_closest(self, n):
        """
        Building a graph connecting each vertex with n closest vertices
        """          
        g_n_closest = self.build_full()
        self._delete_not_closest(g_n_closest, n)
        return g_n_closest



    def build_delaunay(self):
        """
        Building a graph using Delaunay triangulation
        """           
        g_delaunay = self.graph.copy()
        layout = g_delaunay.layout_auto()   
        delaunay = Delaunay(layout.coords)
        for tri in delaunay.simplices:
            g_delaunay.add_edges([
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[0], tri[2]),
            ])
        g_delaunay.simplify()
        
        self._weigh_edges(g_delaunay)
        self._add_segments(g_delaunay, self.edge)
        g_delaunay.es['covered'] = False
        
        self.connected_graph =  g_delaunay
        return g_delaunay
    
    def n_segments(self):
        return sum(self.connected_graph.es["is_segment"])
    
    def total_length(self):
        return sum(self.connected_graph.es["weight"])    
    
    def segment_length(self):
        s_length = 0
        for edge in self.connected_graph.es:
            if edge["is_segment"] == True:    
                s_length += edge["weight"]
        return s_length
            

    def connect_all_to_base(self, g):
        """
        Connect all segment vertices to the base vertex
        """             
        g_gotobase = g.copy()
        for i in range(len(g_gotobase.vs)-1):
            g_gotobase.add_edges([(len(g_gotobase.vs)-1, i)])
        
        g_gotobase.es[ len(g.es): len(g_gotobase.es)]["is_segment"] = False
        g_gotobase.es[ len(g.es): len(g_gotobase.es)]["to_base"] = True
        g_gotobase.es[ 0 : len(g.es)]["to_base"] = False
        return g_gotobase

    def plot_the_graph(self, g):
        """
        Plotting the graph with matplotlib
        """  
        color_dict_vs = {True: "red", False: "black"}
        color_dict_es = {True: "red", False: "black"}
        edge_width = [2 + 10 * int(is_segment) for is_segment in g.es["is_segment"]]
        vertex_color = [color_dict_vs[base] for base in g.vs["base"]]
        edge_color = [color_dict_es[covered] for covered in g.es["covered"]]
        fig, ax = plt.subplots(figsize=(50,50))
        ig.plot(
            g,
            target=ax,
            layout='auto',
            vertex_size = 0.5,
            vertex_label = ["\n     B"* int(base) for base in g.vs["base"]],
            vertex_label_size = 70,
            vertex_frame_width=4.0,
            vertex_color = vertex_color,
            edge_width = edge_width,
            edge_color = edge_color,
            
        )
        plt.show()    
    



