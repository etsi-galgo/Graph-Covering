# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:47:46 2024

@author: Alina
"""
import json
import pandas as pd 
from utils.graph import Graph
import os
from pathlib import Path

#TODO: Change paths

def save_results(q_table, graph, cumulated_reward_dic={}, segments_covered_dic={},coverage_distance_dic={}):
    project = os.path.join(Path.cwd(), 'runs', 'train')
    name='exp'
    log_dir = get_path(project, name)
    
    
    table_file = os.path.join(log_dir, "q_table.json")    
    base_file = os.path.join(log_dir, "base.json")
    line_file = os.path.join(log_dir, "lines.csv")
    cumulated_reward_file = os.path.join(log_dir, "cumulated_reward_dic.json")
    segments_covered_file = os.path.join(log_dir, "segments_covered_dic.json")
    coverage_distance_file = os.path.join(log_dir, "coverage_distance_dic.json")
    
    q_table = dict((str(k), val) for k, val in q_table.items())
    save_dict_to_json(q_table, table_file)
    save_graph(graph, base_file, line_file)
    
    save_dict_to_json(cumulated_reward_dic, cumulated_reward_file)
    save_dict_to_json(segments_covered_dic, segments_covered_file)
    save_dict_to_json(coverage_distance_dic, coverage_distance_file)
    

def open_results(exp):
    project = os.path.join(Path.cwd(), 'runs', 'train')
    log_dir = os.path.join(project, exp)
    table_file = os.path.join(log_dir, "q_table.json")    
    base_file = os.path.join(log_dir, "base.json")
    line_file = os.path.join(log_dir, "lines.csv")
    q_table = get_q_table_from_json(table_file)
    graph = open_graph(base_file, line_file)
    return q_table, graph
    
     

def get_path(project, name, exist_ok=False):
    exp_n = get_exp_n(project, name=name)
    if not exist_ok:
        exp_n += 1
    log_dir = os.path.join(project, "{0}{1}".format(name, exp_n))
    os.makedirs(log_dir, exist_ok=exist_ok)
    return log_dir
    
    
       
def save_dict_to_json(value, filename):
    with open(filename,'w') as outfile:  
          json.dump(value, outfile)
    
    
def get_exp_n(project, name='exp'):
    if not os.path.exists(project):
        return 0
    ns = [
        int(f[len(name):]) for f in sorted(os.listdir(project)) if f.startswith(name) and str.isdigit(f[len(name):])
    ]
    return max(ns) if len(ns) else 0

        
def get_q_table_from_json(filename):
    json_ex = json.load(open(filename))
    return dict((eval(k), val) for k, val in json_ex.items())

def save_graph(graph, base_file, line_file):
    graph.lines.to_csv (line_file, index = False, header=True)
    base_dic = {'base': graph.base}
    save_dict_to_json(base_dic, base_file)
        
def open_graph(base_file, line_file):
    base_dic = json.load(open(base_file))
    base = base_dic['base']
    lines = pd.read_csv(line_file)
    graph = Graph(lines, base)
    return graph