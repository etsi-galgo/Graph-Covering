# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:34:53 2024

@author: Alina Kasiuk
"""

from utils import results, graph_v01
from envs.graph_env import GraphEnv
import qlearn
from sys import platform
import time


def main(mode, env_iter, q_table=None, segment_graph=None, connected_graph=None, total_episodes=1, show=False, alpha=0.3, gamma=0.8, epsilon=0):
    """
    Main function to train or test a Q-learning agent in a graph-based environment.

    Parameters
    ----------
    mode : str
        "train" or "test" mode.
    env_iter : int
        Number of steps permitted every episode.
    q_table : dict, optional
        Previously saved Q-table. Default is None.
    segment_graph : igraph.Graph, optional
        Initial graph with segments. Default is None.
    connected_graph : igraph.Graph, optional
        Graph with Delaunay triangulation. Default is None.
    total_episodes : int
        Number of episodes to run. Default is 1.
    show : bool
        Whether to visualize the environment. Default is False.
    alpha : float
        Learning rate. Default is 0.3.
    gamma : float
        Discount factor. Default is 0.8.
    epsilon : float
        Exploration rate. Default is 0.

    Returns
    -------
    tuple
        Updated graph, Q-table, cumulative reward, coverage percentage, and length covered percentage.
    """
    if segment_graph is None:
        raise ValueError("The segment map is not defined. Cannot continue.")


    if mode == "train":
        eps = epsilon
        epsilon_discount = 0.9986
    elif mode == "test":
        eps = 0
    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")

    ql = qlearn.QLearn(actions=0, alpha=alpha, gamma=gamma, epsilon=eps)
    ql.q = q_table if q_table is not None else {}

    highest_reward = 0

    for episode in range(total_episodes):
        env_ = GraphEnv(connected_graph)
        observation = env_.reset()

        done = False
        cumulated_reward = 0

        if mode == "train" and ql.epsilon > 0.05:
            ql.epsilon *= epsilon_discount

        print(f"Episode {episode + 1}/{total_episodes} - Epsilon: {ql.epsilon:.4f}")

        state = ''.join(map(str, observation))

        for i in range(env_iter):
            if show:
                env_.render()

            node = observation[0]
            n_actions = connected_graph.degree(node)
            ql.actions = range(n_actions)
            action = ql.chooseAction(state)

            observation, reward, done, _ = env_.step(action, node)
            cumulated_reward += reward

            print(f"Step {i} - Action: {action}, Reward: {reward}")

            if cumulated_reward > highest_reward:
                highest_reward = cumulated_reward

            next_state = ''.join(map(str, observation))
            print(f"State: {next_state}")

            ql.learn(state, action, reward, next_state)

            if done:
                break

            state = next_state

        print(f"Cumulated reward: {cumulated_reward}")
 #       print(f"Segments covered: {env_.n_segments_covered} of {g.n_segments}")
 #       print(f"Length covered: {env_.coverage_distance} of {segment_length}")

        env_.close()

    return ql.q
#, cumulated_reward, env_.n_segments_covered, env_.coverage_distance


if __name__ == "__main__":
    mode = "train"
    show = platform == "win32"

    if mode == "train":
        # Training the model
        n = 10
        graph = graph_v01.MyGraph(width=100, height=100, n_lines=5, max_segs_per_line=4, base=[0, 25])
        segment_graph=graph.segment_graph

        # Generate the Delaunay triangulation graph for connectivity
        connected_graph = graph.delaunay_graph()
        

        # Train the model for the first iteration
        q_table = main(mode, env_iter=200, segment_graph=segment_graph, connected_graph=connected_graph, total_episodes=10, show=show)

        # Continue training in subsequent iterations
        for i in range(n - 1):
            q_table = main(mode, env_iter=200, q_table=q_table, segment_graph=segment_graph, connected_graph=connected_graph, total_episodes=10, show=show)

#        results.save_results(q_table, graph)
#TODO: save graph as a vertex list

    elif mode == "test":
        # Testing the result
        q_table, saved_graph = results.open_results("exp5")

        # Recreate the graph object from saved data
        graph = graph_v01.MyGraph(given_graph=saved_graph.segment_graph)
        graph.connected_graph = graph.delaunay_graph()

        main(mode, env_iter=200, q_table=q_table, segment_graph=graph.segment_graph, connected_graph=graph.connected_graph, total_episodes=200, show=show)

    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")

    
    
