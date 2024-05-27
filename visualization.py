# -*- coding: utf-8 -*-

import json 
import os 
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

#Argumentos funcion
alpha_values = [0.01,0.05,0.1,0.15,0.2,0.3]
gamma_values = [0.99,0.95,0.9,0.85,0.8]
epsilon_values = [(1+i)*0.05 for i in range(0,19)]
metrica = "coverage_distance_dic" # o "cumulated_reward_dic" o "segments_covered_dic"
metrica = "cumulated_reward_dic"
metrica = "segments_covered_dic"
######

def graph_2d(alpha_values,gamma_values,epsilon_values,metrica,fijar,valor_fijo):
    #Ahora  se fijan dos hiperparametros 
    
    #Working directory should be Graph-Covering
    path = os.getcwd()+"\\runs\\hyperparameters_study\\"
    
    with open(path+metrica+'.json') as json_file:
        coverage_distance_dic = json.load(json_file)
    
    #hiperparametros de estudio 
    hyperparam = ["alpha","gamma","epsilon"]
    #Metemos valores en diccionario 
    dic = {"alpha":alpha_values,"gamma":gamma_values,"epsilon":epsilon_values}
    
    
    if valor_fijo[0] not in dic[fijar[0]] or valor_fijo[1] not in dic[fijar[1]]:
        raise Exception("Valor fijo no se usa en el hiperparametro indicado, asegurate de que  ese valor se usa en el hiperparametro")
        
 #incides asociados a los hiperparametros que se usan para la base 

    for key in dic:
        if key == fijar[0] or key == fijar[1]: 
            continue 
        index = hyperparam.index(key) #indice del eje x
        grid = dic[key] #valores del  eje 
    

    
    values = np.zeros(shape=len(grid)) #valores de la metrica 
    
    for key in coverage_distance_dic: 
        key_tuple = eval(key) #str --> tuple
        if key_tuple[hyperparam.index(fijar[0])]==valor_fijo[0] and key_tuple[hyperparam.index(fijar[1])]==valor_fijo[1] :
            indice_valor = grid.index(key_tuple[index]) #cogemos la posicion del valor en el grid 
            values[indice_valor] = coverage_distance_dic[key]
        
    

    fig,axes = plt.subplots()
    axes.plot(grid,values, color ='green')
    axes.set_title(metrica+" "+fijar[0]+" = "+ str(valor_fijo[0])+" y "+fijar[1]+" = "+str(valor_fijo[1]))
    axes.set_xlabel(hyperparam[index])
    plt.show()

#Lanzamos 
graph_2d(alpha_values, gamma_values, epsilon_values, metrica, ["epsilon","gamma"], [0.4,0.8])


def graph_3d(alpha_values,gamma_values,epsilon_values,metrica,fijar,valor_fijo):
    
    #Working directory should be Graph-Covering
    path = os.getcwd()+"\\runs\\hyperparameters_study\\"
    
    with open(path+metrica+'.json') as json_file:
        coverage_distance_dic = json.load(json_file)
    
    #hiperparametros de estudio 
    hyperparam = ["alpha","gamma","epsilon"]
    #Metemos valores en diccionario 
    dic = {"alpha":alpha_values,"gamma":gamma_values,"epsilon":epsilon_values}
    
    
    if valor_fijo not in dic[fijar]:
        raise Exception("Valor fijo no se usa en el hiperparametro indicado, asegurate de que  ese valor se usa en el hiperparametro")
        
    indexes=[] #incides asociados a los hiperparametros que se usan para la base 
    grid = []
    for key in dic:
        if key == fijar: 
            continue 
        indexes.append(hyperparam.index(key))
        grid.append(dic[key])
    
    print(indexes)
    X,Y = np.meshgrid(grid[0],grid[1])
    
    
    
    
    
    value = np.zeros(X.shape)
    
    for key in coverage_distance_dic: 
        key_tuple = eval(key) #str --> tuple
        if key_tuple[hyperparam.index(fijar)]==valor_fijo:
            value[grid[1].index(key_tuple[indexes[1]]),grid[0].index(key_tuple[indexes[0]])]= coverage_distance_dic[key]
        
    
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(X, Y, value, color ='green')
    ax.set_title(metrica+" "+fijar+" = "+ str(valor_fijo))
    ax.set_xlabel(hyperparam[indexes[0]],fontsize = 20)
    ax.set_ylabel(hyperparam[indexes[1]],fontsize = 20)
    plt.show()


fijar = "alpha" 
valor_fijo= 0.01 #cuidado, debe ser un valor de algun hiperparametro
#######

graph_3d(alpha_values, gamma_values, epsilon_values, metrica, fijar, valor_fijo)