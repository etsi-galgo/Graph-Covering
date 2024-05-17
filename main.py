# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

from utils.graph import Graph
from utils import results
from envs.graph_env import GraphEnv
import qlearn
from sys import platform




def main(mode, env_iter, q_table={}, g = Graph(), total_episodes=1,  show = False):
    """
    Parameters
    ----------
    mode : "train" or "test" mode
    env_iter : Number of steps permited every episode
    q_table : dictionary. Introduce if continue with a previosly saved Q-table. 
    The default is {} if starting a new training.
    g : a custom Graph object. Introduce if continue with a previosly saved graph. 
        The default is Graph() means create a new graph.
    total_episodes : How many times reset the environment
    show : Visualization. 

    Returns
    -------
    g: graph
    ql.q: Q-table

    """
    if mode=="train":
        eps=0.9
        epsilon_discount = 0.9986
    elif mode=="test":
        eps=0
    else:
        print("Unknown mode")
        return
        
    ql = qlearn.QLearn(actions = 0, alpha=0.2, gamma=0.8, epsilon=eps)
    ql.q = q_table
    
    
    highest_reward = 0
    for episode in range(total_episodes):
        print("Episode ", episode, " of ", total_episodes)
        
        #TODO: put this out of cycle. Careful with segments:
        g.build_delaunay() #Recover a graph
        
        env_ = GraphEnv(g.connected_graph) #Reset the environment
        observation = env_.reset()
        
        #TODO: put this out of cycle:
        s_N = g.n_segments() #Count number of segments
        s_length = g.segment_length() #Count total lenght to cover
        
        #Initialize counters:
        done = False 
        cumulated_reward = 0
        
        #Every episode trust more to the model: 
        if ql.epsilon > 0.05:
            ql.epsilon *= epsilon_discount
            
        #State as a string for the q-table dictionary    
        state = ''.join(map(str, observation))

        #Apply env_iter steps or until done:
        for i in range(env_iter): 
            if show:
                env_.render() #Visualization
            
            #Pick an action based on the current state:
            node = observation[0]
            n_actions = g.connected_graph.degree(node)
            ql.actions = range(n_actions)
            action = ql.chooseAction(state)
            
            #Execute the action and get feedback:                    
            observation, reward, done, _ = env_.step(action, node)
            cumulated_reward += reward
            
            
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                
            nextState = ''.join(map(str, observation))   
            
            #Update the Q-table:
            ql.learn(state, action, reward, nextState)
   
            if not(done):
                state = nextState
            else:
                #Done if all segments are covered or the battery is over
                break
            
                
        print("Cumulated reward:", cumulated_reward)
        print("Segments covered:", env_.n_segments_covered, "of", s_N)
        print("Length covered:", env_.coverage_distance, "of",  s_length)
        
        env_.close()
        
    print("Highest reward:", highest_reward)
    return g, ql.q

    
if __name__ == "__main__":
    mode="test"
    
    if platform == "win32": show = True
    else: show = False
    
    if mode=="train": #Training the model
        graph, q_table = main(mode, env_iter=200, total_episodes=5,  show = show)
        results.save_results(q_table, graph) #Save
        
    if mode=="test": #Testing the result 
        q_table, graph = results.open_results("exp1")
        main(mode, env_iter=200, q_table = q_table, g = graph, show = show)

    
    
    
