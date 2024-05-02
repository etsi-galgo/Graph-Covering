# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

from utils.graph import Graph
from envs.graph_env import GraphEnv
import qlearn
from sys import platform
import json
import pandas as pd

def main(total_episodes, env_iter, show = False):
    g = Graph()
    
    ql = qlearn.QLearn(actions = 0, alpha=0.2, gamma=0.8, epsilon=0.9)
    epsilon_discount = 0.9986
    highest_reward = 0
        
    for episode in range(total_episodes):
        graph_path = g.build_delaunay()
        env_ = GraphEnv(graph_path)
        s_N = g.n_segments()
        s_length = g.segment_length()
        
        done = False
        cumulated_reward = 0
        
        node, observation = env_.reset()
        
        if ql.epsilon > 0.05:
            ql.epsilon *= epsilon_discount
            
        state = ''.join(map(str, observation))
            
            

        print("Episode ", episode, " of ", total_episodes)
        
        for i in range(env_iter):
            if show:
                env_.render()
            n_actions = graph_path.degree(node)

            # Pick an action based on the current state
            ql.actions = range(n_actions)
            action, q = ql.chooseAction(state, return_q = True)
            
            # Execute the action and get feedback                    
            node, observation, reward, done, _ = env_.step(action, node)
            
            cumulated_reward += reward
            
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                
            nextState = ''.join(map(str, observation))    
            ql.learn(state, action, reward, nextState)
   
            if not(done):
                state = nextState
            else:
                break
        print(i)
                
        print("Cumulated reward:", cumulated_reward)
        print("Traveled distance:", env_.total_traveled_distance)
        print("Total length:",  g.total_length())
        print("Segments covered:", env_.n_segments_covered, "of", s_N)
        print("Length covered:", env_.coverage_distance, "of",  s_length)
        
        
        env_.close()
    return g, ql.q

def test(q_table, graph, env_iter, show = True):
    
    ql = qlearn.QLearn(actions = 0, alpha=0.2, gamma=0.8, epsilon=0)
    ql.q = q_table
    
    highest_reward = 0
        
    graph_path = graph.build_delaunay()
    env_ = GraphEnv(graph_path)
    s_N = graph.n_segments()
    s_length = graph.segment_length()
        
    done = False
    cumulated_reward = 0
        
    node, observation = env_.reset()
        
            
    state = ''.join(map(str, observation))
            
        
    for i in range(env_iter):
        if show:
            env_.render()
        n_actions = graph_path.degree(node)

        # Pick an action based on the current state
        ql.actions = range(n_actions)
        action = ql.chooseAction(state)
        print(action)
        # Execute the action and get feedback                    
        node, observation, reward, done, _ = env_.step(action, node)
        cumulated_reward += reward
            
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward
                
        nextState = ''.join(map(str, observation))    
        ql.learn(state, action, reward, nextState)
   
        if not(done):
            state = nextState
        else:
            break
        
    print(i)        
    print("Cumulated reward:", cumulated_reward)
    print("Traveled distance:", env_.total_traveled_distance)
    print("Total length:",  graph.total_length())
    print("Segments covered:", env_.n_segments_covered, "of", s_N)
    print("Length covered:", env_.coverage_distance, "of",  s_length)     
        
    env_.close()
    
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
     
    
    
if __name__ == "__main__":

    total_episodes = 500
    env_iter = 200
    
    if platform == "win32": show = True
    else: show = False
    # or vars(opt)
#    graph, q_table = main(total_episodes, env_iter, show)
#    save_q_table_to_json(q_table)
#    save_graph(graph)
    
    # test
    # recover Q-table
    q_table = get_q_table_from_json()
    # recover graph
    graph = open_graph()
    
    test(q_table, graph, env_iter)
    
    
    
