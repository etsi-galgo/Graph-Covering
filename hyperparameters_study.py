# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:33:36 2024

@author: aureb
"""

# from utils.graph import Graph
# from utils import results
# from envs.graph_env import GraphEnv
# import qlearn
# from sys import platform
# import timeit
from main import main 
from utils import results

# alpha_values = [0.01,0.05,0.1,0.15,0.2,0.3]
# gamma_values = [0.99,0.95,0.9,0.85,0.8]
#epsilon_values = [i/20 for i in range(1,19)]
#print(epsilon_values)


alpha_values = [0.01,0.05]
gamma_values = [0.99,0.95]
epsilon_values = [round((1+i)*0.05,2) for i in range(0,5)]


#Guardamos los datos del estudio en los siguientes diccionarios
cumulated_reward_dic={} #Diccionario (alpha,gamma,epsilon):comulated_reward
segments_covered_dic={} #Diccionario (alpha,gamma,epsilon):segments_covered
coverage_distance_dic={} #Diccionario (alpha,gamma,epsilon):coverage_distance


contador_j = 0


print("En cada paso recorremos todos los epsilon en epsilon_values. Duración aproximada por paso: ", len(epsilon_values), "minutos")
for i in alpha_values:

    for j in gamma_values:
        contador_parcial_k = 0
        contador_j = contador_j+1
        print("Paso ",contador_j," de", len(alpha_values)*len(gamma_values))
        for k in epsilon_values:
            contador_parcial_k = contador_parcial_k + 1
            graph, q_table,cumulated_reward,segments_covered,coverage_distance = main("train", 0, 1, env_iter=200, total_episodes=1000,  show = False, alpha = i,gamma=j,epsilon=k)
            graph, q_table,cumulated_reward,segments_covered,coverage_distance = main("test", 0, 1, env_iter=200, q_table=q_table, g = graph, total_episodes=1,  show = False, alpha = i,gamma=j,epsilon=k)
            cumulated_reward_dic[(i,j,k)]=cumulated_reward
            segments_covered_dic[(i,j,k)]=segments_covered
            coverage_distance_dic[(i,j,k)]=coverage_distance
            
            print("    ", contador_parcial_k/len(epsilon_values)*100, "%")


#Guardado de datos 
results.save_results(q_table, graph, cumulated_reward_dic, segments_covered_dic, coverage_distance_dic) #Save

   
# #Lectura de datos 
# import ast
# with open('cumulated_reward_dic.txt') as f:  
#       cumulated_reward_dic = f.read()
# cumulated_reward_dic = ast.literal_eval(cumulated_reward_dic) #pasamos a tipo diccionario
# with open('segments_covered_dic.txt') as f:  
#       segments_covered_dic = f.read()
# segments_covered_dic = ast.literal_eval(segments_covered_dic)
# with open('coverage_distance_dic.txt') as f:  
#       coverage_distance_dic = f.read()
# coverage_distance_dic = ast.literal_eval(coverage_distance_dic)
 

            
      
            
            