# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:47:46 2024

@author: Alina
"""
import json
import pandas as pd 
from utils.graph import Graph

#TODO: Change paths
def save_q_table_to_json(q_table, filename = "sample.json"):
    q_table2 = dict((str(k), val) for k, val in q_table.items())
    with open(filename, "w") as outfile: 
        json.dump(q_table2, outfile)
        
def get_q_table_from_json(filename = "sample.json"):
    json_ex = json.load(open("sample.json"))
    return dict((eval(k), val) for k, val in json_ex.items())

def save_graph(graph):
    graph.lines.to_csv ("lines.csv", index = False, header=True)
    base_dic = {'base': graph.base}
    with open("base.json", "w") as outfile: 
        json.dump(base_dic, outfile)   
        
def open_graph(base_file = "base.json", line_file = "lines.csv"):
    base_dic = json.load(open("base.json"))
    base = base_dic['base']
    lines = pd.read_csv("lines.csv")
    graph = Graph(lines, base)
    return graph