# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

import argparse
from utils import graph_v01
from envs.graph_env import GraphEnv
import qlearn
from utils.results import save_results, open_results
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_opt():
    """
    Function to parse command-line options.
    """
    parser = argparse.ArgumentParser(description="Train or test a Q-learning agent in a graph environment")
    
    # Mode and general settings
    parser.add_argument('--mode', type=str, default='train', choices=['train','train_more', 'test'], help="Mode: 'train' or 'test'")
    parser.add_argument('--experiment', type=str, default='exp1', help="Experiment file name to load")
    parser.add_argument('--env-iter', type=int, default=200, help="Number of iterations allowed per episode")
    parser.add_argument('--total-episodes', type=int, default=10000, help="Total number of episodes")
    parser.add_argument('--alpha', type=float, default=0.3, help="Learning rate for Q-learning")
    parser.add_argument('--gamma', type=float, default=0.8, help="Discount factor for Q-learning")
    parser.add_argument('--epsilon', type=float, default=1, help="Exploration rate (epsilon) for Q-learning")
    parser.add_argument('--epsilon-discount', type=float, default=0.9997, help="Discount for epsilon during the training")
    parser.add_argument('--show', action='store_true', help="Visualize the environment during execution")

    
    
    # Graph settings
    parser.add_argument('--width', type=int, default=100, help="Graph width")
    parser.add_argument('--height', type=int, default=100, help="Graph height")
    parser.add_argument('--n-lines', type=int, default=5, help="Number of lines in the graph")
    parser.add_argument('--max-segs-per-line', type=int, default=4, help="Maximum number of segments per line")
    parser.add_argument('--base', type=int, nargs=2, default=[0, 25], help="Base station coordinates")

    return parser.parse_args()

def main(train_vars):
    """
    Main function to train or test a Q-learning agent in a graph-based environment.

    Parameters
    ----------
    train_vars : dict
        Dictionary containing all training/testing parameters.
    """
    mode = train_vars['mode']
    total_episodes = train_vars['total_episodes']
    env_iter = train_vars['env_iter']
    show = train_vars['show']
    alpha = train_vars['alpha']
    gamma = train_vars['gamma']
    epsilon = train_vars['epsilon']
    epsilon_discount = train_vars['epsilon_discount']
    
    base = train_vars['base']
    width = train_vars['width']
    height = train_vars['height']
    n_lines = train_vars['n_lines']
    max_segs_per_line = train_vars['max_segs_per_line']
    exp = train_vars['experiment']
    
    if mode == "train":
        q_table = {}
        # Create new graph
        graph = graph_v01.MyGraph(None, base, width, height, n_lines, max_segs_per_line)        
        
    elif mode == "train_more":
        q_table, vertices = open_results(exp)
        graph = graph_v01.MyGraph(vertices, base)
        
    elif mode == "test":
        epsilon = 0  # Disable exploration during testing
        total_episodes = 5
        q_table, vertices = open_results(exp)
        graph = graph_v01.MyGraph(vertices, base)
    
    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")
    
    
    
    # Initialize Q-learning agent with given parameters or a blank Q-table if not provided
    ql = qlearn.QLearn(actions=0, alpha=alpha, gamma=gamma, epsilon=epsilon)
    ql.q = q_table  # Initialize empty Q-table (or load it if you're continuing training)
    
    highest_reward = 0  # Track the highest reward achieved during episodes
    
    max_change_threshold = 1e-15  # Define a threshold for convergence
    last_q_values = ql.q.copy()  # Store last Q-table values for comparison
    rewards_list = []
    
    # Iterate over the number of episodes
    for episode in tqdm(range(total_episodes)):
        connected_graph = graph.delaunay_graph()
        env_ = GraphEnv(connected_graph, ql)  # Initialize environment with the connected graph
        observation = env_.reset()  # Reset the environment for each episode
        done = False  # Flag for episode termination
        cumulated_reward = 0  # Track cumulative reward per episode
        
        
        # Decay epsilon during training if it's greater than a minimum threshold
        if ql.epsilon > 0.05:
            ql.epsilon *= epsilon_discount
            
        # Print progress for the current episode
        print(f"Episode {episode + 1}/{total_episodes} - Epsilon: {ql.epsilon:.4f}")
        
        # Convert observation into a string state representation
        state = ''.join(map(str, observation)) 
        
        # Step through each episode for the given number of iterations (env_iter)
        for i in range(env_iter):
            if show:
                env_.render()  # Optionally visualize the environment

            node = observation[0]  # Get the current node from the observation
            n_actions = connected_graph.degree(node)  # Number of actions based on node degree
            ql.actions = range(n_actions)  # Define available actions

            # Choose an action using the Q-learning policy (either exploration or exploitation)
            action = ql.chooseAction(state)

            # Perform the chosen action and receive the new state, reward, and completion status
            observation, reward, done, _ = env_.step(action, node)
            cumulated_reward += reward  # Update cumulative reward
            # Inside your episode loop
            
            
            if mode == "test":
                print(f"Step {i} - Action: {action}, Reward: {reward}")
                print(f"State: {observation}")

            # Track the highest reward achieved so far
            if cumulated_reward > highest_reward:
                highest_reward = cumulated_reward

            next_state = ''.join(map(str, observation))  # Convert next observation into a string state
            

            # Update Q-table using the Q-learning algorithm
            if not mode == "test":
                ql.learn(state, action, reward, next_state)

            # If the episode is done (reached a terminal state), exit the loop
            if done:
                print(i)
                break

            state = next_state  # Move to the next state for the next iteration
            
        seg_left = sum(env_.graph.es["is_segment"])
        rewards_list.append(cumulated_reward)
        n_seg = graph.n_segments
        print(f"Segments left: {seg_left} / {n_seg}")   
        print(f"Cumulated reward: {cumulated_reward}")   
        env_.close()  # Clean up the environment at the end of each episode
        
        max_change = max(abs(last_q_values.get((state, action), 0) - ql.q.get((state, action), 0))
                         for action in ql.actions)

        if max_change < max_change_threshold:
            print(f"Q-table has converged at episode {episode + 1}.")

        last_q_values = ql.q.copy()  # Update for next comparison
        
    # Safe vertices
    vertices = graph.vertices
    # Save the results after training/testing
    if not mode == "test":
        save_results(ql.q, vertices, cumulated_reward_dic={"episode": episode + 1, "reward": cumulated_reward},
                     segments_covered_dic={}, coverage_distance_dic={})

    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress')
    plt.show()
        

if __name__ == "__main__":
    opt = parse_opt()  # Parse arguments from the command line
    for i in range(10):
        main(vars(opt))  # Pass the arguments as a dictionary to the main function
