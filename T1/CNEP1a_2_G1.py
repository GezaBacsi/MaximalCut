#%%
import ctypes
import multiprocessing
import os
import sys
import random

import matplotlib.pyplot as plt

import networkx as nx
from networkx.generators.trees import NIL
from networkx.readwrite import edgelist
from networkx.algorithms import approximation as approx
from networkx import Graph as gr

import itertools

import numpy as np

## -> MultiProcessing
from multiprocessing import *
pool_size = 5
## <- MultiProcessing

print("X")

inputFile = "Teszt\\teszt1.txt"
#inputFile = "URV\\email.txt"
G = nx.read_adjlist(sys.path[0]+"\\..\\..\\Grafok\\" + inputFile, nodetype = int)

print("G - edgecount: " + str(len(G.edges)))
print("G - nodecount: " + str(len(G.nodes)))

# plt.subplot(121)
# nx.draw(G)
plt.subplot(131)
nx.draw(G, with_labels=True) #, pos=nx.planar_layout(G), node_color='r', edge_color='b')

# %% 
# CNP 1a G1
# INIT VALUES
K1 = 2 # vertices to delete
K2 = 2 # edges to delete
K = K1 + K2
INF = G.number_of_nodes() ** 2
IterationCount = G.number_of_nodes() ** 2

print("0")
manager = multiprocessing.Manager()
print("1")
minVal = manager.Value('i',1, lock=True)
minVal = 0
minR = NIL
minSS = manager.list()
minEE = manager.list()
print("AAA")

def f_pairwise(lst): ### A komponensek szamossagabol a "pairwise-connectivity"
    summa = sum([len(c)*(len(c)-1)/2 if (len(c)>1) else 0 for c in lst])
    return summa

def h_component(lst): ### A komponensek szamossagabol a "pairwise-connectivity"
    return sum(lst)

def select_random(lst): ### Veletlen elem
    return random.choice(lst)


#%%

#%%
def best_nodes_edges_CNEP1A_Alg2(G,SN,SE): ### Legjobb pontok es elek kigeneralasa
    selectedEdges = []
    selectedNodes = []
    min_pw = INF
    P = G.copy()
    P.remove_nodes_from(SN)
    P.remove_edges_from(SE)
    node_f_orig = f_pairwise(nx.connected_components(P))
    SG2 = nx.edges(G)
    SG1 = nx.nodes(G)
    #print("--------------------------------------------\nKezdes")
    #print("-> S: ", S)
    #print("   node_f_orig = ", node_f_orig)
    if len(SN) < K1:
        for curr_node in SG1:
            R = P.copy()
            R.remove_nodes_from([curr_node])
            node_f = node_f_orig - f_pairwise(nx.connected_components(R))
            #print("      node_f = ", node_f," (S: ",set(S)-set([curr_node]),")")

            if node_f < min_pw:
                selectedNodes.clear
                selectedNodes.append(curr_node)
                min_pw = node_f
            elif node_f == min_pw:
                selectedNodes.append(curr_node)

    if len(SE) < K2:
         for curr_edge in SG2:
            R = P.copy()
            R.remove_edges_from([curr_edge])
            node_f = node_f_orig - f_pairwise(nx.connected_components(R))
            #print("      node_f = ", node_f," (S: ",set(S)-set([curr_node]),")")

            if node_f < min_pw:
                selectedEdges.clear
                selectedEdges.append(curr_edge)
                min_pw = node_f
            elif node_f == min_pw:
                selectedEdges.append(curr_edge)
    #print("->N:",selectedNodes)   
    #print("->E:",selectedEdges)   
    return [selectedNodes,selectedEdges]

# CNP1a Alg2 G2
def CNEP1a_2_G1(G):
    S = []
    E = []
    
    while len(S)+len(E) < K:
        [A,B] = best_nodes_edges_CNEP1A_Alg2(G,S,E)
        #print("B: ", B)
        z1 = z2 = NIL
        if len(A)>0:
            z1 = select_random(A)
            #print("--> (randN)",z1)
        if len(B)>0:
            z2 = select_random(B)
            #print("--> (randE)",z2)

        if (z1!=NIL):
            if (z2!=NIL):
                if random.randint(0,1) == 1:
                    S.append(z1)
                    G.remove_nodes_from([z1])
                    #print("--> (del)N")
                else:
                    E.append(z2)
                    G.remove_edges_from([z2])
                    #print("--> (del)E")
            else:
                S.append(z1)
                G.remove_nodes_from([z1])
                #print("--> (del)N")
        else:
            E.append(z2)
            G.remove_edges_from([z2])
            #print("--> (del)E")

    H = G.copy()
    H.remove_nodes_from(S)
    H.remove_edges_from(E)
    
    return [H,S,E]

#%%

def makeCNEPRun(G,method):
    global minVal
    global minSS
    global minEE
    print(".")
    [R,SS, EE] = method(G)
    #print(SS)
    currVal = f_pairwise(nx.connected_components(R))
    #print(currVal)
    if currVal<minVal:
        minVal = currVal
        #minR = R
        minSS = SS
        minEE = EE

#%%

######## 
# RunAndPlotCNEPMethod(G,CNEP1a_2_G1,133)
########

if __name__ == '__main__':
    method = CNEP1a_2_G1
    plotNo = 111

    pool = Pool(pool_size)
    print("IterationCount: "+str(IterationCount))
    for x in range(1,IterationCount):
        print(str(x)+": begin")
        pool.apply(makeCNEPRun,args=(G, CNEP1a_2_G1,))
        print(str(x)+": end")

    pool.close()
    pool.join()
    
    print(method.__name__)
    
    print(minSS)
    print(minEE)
    print(minVal)
    plt.subplot(plotNo)
    minR = G.copy()
    minR.remove_nodes_from(minSS)
    minR.remove_nodes_from(minEE)
    nx.draw(minR, with_labels=True) #, pos=nx.planar_layout(G), node_color='r', edge_color='b')


# %%