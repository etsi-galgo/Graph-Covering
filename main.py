# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

from utils.graph import Graph
from envs.graph_env import GraphEnv
import qlearn

def main(total_episodes, env_iter):
    g = Graph()
    graph_path = g.build_delaunay()
    env_ = GraphEnv(graph_path)
    
    ql = qlearn.QLearn(actions = 0, alpha=0.2, gamma=0.8, epsilon=0.9)
    epsilon_discount = 0.9986
    highest_reward = 0
        
    for episode in range(total_episodes):
        done = False
        cumulated_reward = 0
        
        node, observation = env_.reset()
        
        if ql.epsilon > 0.05:
            ql.epsilon *= epsilon_discount
            
        state = ''.join(map(str, observation))
            
            

        print("Episode ", episode, " of ", total_episodes)
        
        for i in range(env_iter):
            env_.render()
            n_actions = graph_path.degree(node)

            # Pick an action based on the current state
            ql.actions = range(n_actions)
            action = ql.chooseAction(state)
            
            # Execute the action and get feedback                    
            node, observation, reward, done, _ = env_.step(action, node)
            cumulated_reward += reward
            
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                
            nextState = ''.join(map(str, observation))    
            ql.learn(state, action, reward, nextState)
   
            if not(done):
                state = nextState
   
    env_.close()
    
if __name__ == "__main__":

    total_episodes = 5
    env_iter = 20
    
    # or vars(opt)
    main(total_episodes, env_iter)
