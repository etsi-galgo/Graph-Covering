# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:47:46 2024

@author: Alina
"""
import json
import numpy as np
import os
from pathlib import Path

def save_results(q_table, vertices, cumulated_reward_dic={}, segments_covered_dic={}, coverage_distance_dic={}):
    """
    Save Q-table, segment vertices and training performance to files.
    """
    project = os.path.join(Path.cwd(), 'runs', 'train')
    name = 'exp'
    log_dir = get_path(project, name)
    
    # Define file paths
    table_file = os.path.join(log_dir, "q_table.json")    
    vertices_file = os.path.join(log_dir, "vertices.txt")
    cumulated_reward_file = os.path.join(log_dir, "cumulated_reward_dic.json")
    segments_covered_file = os.path.join(log_dir, "segments_covered_dic.json")
    coverage_distance_file = os.path.join(log_dir, "coverage_distance_dic.json")

    try:
        save_dict_to_json(q_table, table_file)
        save_vertices(vertices, vertices_file)
        save_dict_to_json(cumulated_reward_dic, cumulated_reward_file)
        save_dict_to_json(segments_covered_dic, segments_covered_file)
        save_dict_to_json(coverage_distance_dic, coverage_distance_file)
    except Exception as e:
        print(f"Error saving results: {e}")

def open_results(exp):
    """
    Load segment vertices and Q-table from files.
    """
    project = os.path.join(Path.cwd(), 'runs', 'train')
    log_dir = os.path.join(project, exp)
    table_file = os.path.join(log_dir, "q_table.json")    
    vertices_file = os.path.join(log_dir, "vertices.txt")

    try:
        q_table = get_dict_from_json(table_file)
        vertices = load_vertices(vertices_file)
        return q_table, vertices
    except Exception as e:
        print(f"Error opening results: {e}")
        return None, None

def get_path(project, name, exist_ok=False):
    """
    Get or create the directory path for saving results.
    """
    exp_n = get_exp_n(project, name=name)
    if not exist_ok:
        exp_n += 1
    log_dir = os.path.join(project, "{0}{1}".format(name, exp_n))
    os.makedirs(log_dir, exist_ok=exist_ok)
    return log_dir

def save_dict_to_json(value, filename):
    """
    Save a dictionary to a JSON file.
    """
    try:
        str_keys_dic = dict((str(k), val) for k, val in value.items())
        with open(filename, 'w') as outfile:  
            json.dump(str_keys_dic, outfile)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        
        
def get_dict_from_json(filename):
    """
    Load a dictionary from a JSON file.
    """
    try:
        with open(filename, 'r') as file:
            json_ex = json.load(file)
        return dict((eval(k), val) for k, val in json_ex.items())
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

def get_exp_n(project, name='exp'):
    """
    Get the next experiment number based on existing directories
    """
    if not os.path.exists(project):
        return 0
    ns = [
        int(f[len(name):]) for f in sorted(os.listdir(project)) if f.startswith(name) and str.isdigit(f[len(name):])
    ]
    return max(ns) if len(ns) else 0


def save_vertices(vertices, filename):
    """
    Save vertices to a text file.
    """
    try:
        np.savetxt(filename, vertices, fmt='%d', delimiter=',', header='x,y', comments='')
    except Exception as e:
        print(f"Error saving vertices: {e}")

def load_vertices(filename):
    """
    Load vertices from a text file.
    """
    try:
        return np.loadtxt(filename, delimiter=',', skiprows=1)  # Adjust skiprows if there is no header
    except Exception as e:
        print(f"Error loading vertices: {e}")
        return np.array([])
