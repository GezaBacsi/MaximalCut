#%%
import os
import sys
import random
import networkx as nx
from networkx.algorithms import approximation as approx
from networkx import Graph as gr
import deap

import matplotlib.pyplot as plt
from networkx.generators.trees import NIL

import itertools

from networkx.readwrite import edgelist

import numpy as np
from numpy.lib.function_base import append

inputFile = "Teszt\\teszt1.txt"

G = nx.read_adjlist(sys.path[0]+"\\..\\..\\Grafok\\" + inputFile, nodetype = int)

# plt.subplot(121)
# nx.draw(G)
# plt.subplot(122)
# nx.draw(G, with_labels=True) #, pos=nx.planar_layout(G), node_color='r', edge_color='b')

# %% 
# CNP 1a G1
# INIT VALUES
K1 = 2 # vertices to delete
K2 = 2 # edges to delete
K = K1 + K2
INF = G.number_of_nodes() ** 2
IterationCount = G.number_of_nodes() ** 2

#%%

def genObjectiveF(G,status, eList, nList): ### Legjobb pontok es elek kigeneralasa
    selectedEdges = []
    selectedNodes = []
    for i in range(0,len(status[0])-1):
        if status[0][i] == 0:
            selectedEdges.append(eList[i])

    for i in range(0,len(status[1])-1):
        if status[1][i] == 0:
            selectedEdges.append(nList[i])

    P = G.copy()
    P.remove_nodes_from(selectedEdges)
    P.remove_edges_from(selectedNodes)
    return f_pairwise(nx.connected_components(P))

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2, r_cross, retry_count):
	# children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
		# select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-2)
        counter = int(retry_count*len(p1))
        while sum(p1[:pt]) != sum(p2[:pt]) and counter:
            pt = np.random.randint(1, len(p1)-2)
            counter = retry_count - 1
        # perform crossover
        if counter:
            c1 = list(p1[:pt]) + list(p2[pt:])
            c2 = list(p2[:pt]) + list(p1[pt:])
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    mut_count = 0
    for i in range(len(bitstring)):
		# check for a mutation
        if np.random.rand() < r_mut:
			# flip the bit
            bitstring[i] = 1 - bitstring[i]
            # az uj nullasok szamat nyilvantartom 
            if bitstring[i] == 0:
                mut_count = mut_count + 1
            else:
                mut_count = mut_count - 1
    while mut_count > 0:
        i = np.random.randint(0, len(bitstring)-1)
        if bitstring[i] == 0:
            bitstring[i] = 1 - bitstring[i]
            mut_count = mut_count - 1
            
    while mut_count < 0:
        i = np.random.randint(0, len(bitstring)-1)
        if bitstring[i] == 1:
            bitstring[i] = 1 - bitstring[i]
            mut_count = mut_count + 1
            


# genetic algorithm
def genetic_algorithm(objective, K1, K2, n_iter, n_pop, r_cross, r_mut, retry_count):
	# initial population
    eList = list(G.edges)
    z1 = list(itertools.repeat(0,K1))+list(itertools.repeat(1,len(eList)-K1))
    nList = list(G.nodes)
    z2 = list(itertools.repeat(0,K2))+list(itertools.repeat(1,len(nList)-K2))
    
    pop = [[np.random.permutation(z1),np.random.permutation(z2)] for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(G,pop[0],eList,nList)
	# enumerate generations
    for gen in range(n_iter):
		# evaluate all candidates in the population
        scores = [objective(G,c,eList,nList) for c in pop]
		# check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs 
            p1, p2 = selected[i], selected[i+1]
            
            # crossover and mutation
            c1a, c2a = crossover(p1[0], p2[0], r_cross, retry_count)
            c1b, c2b = crossover(p1[1], p2[1], r_cross, retry_count)
            mutation(c1a,r_mut)
            mutation(c1b,r_mut)
            cc = [c1a, c1b]
            children.append(cc)
            mutation(c2a,r_mut)
            mutation(c2b,r_mut)
            cc = [c2a, c2b]
            children.append(cc)
            # store for next generation
		# replace population
        pop = children

    return [best, best_eval]

n_pop = 10
n_iter = 10
r_crossover = 0.2
r_mutate = 0.3
r_retry = 0.5

print(genetic_algorithm(genObjectiveF,K1,K2,n_iter,n_pop,r_crossover,r_mutate,r_retry))

#%%

def f_pairwise(lst): ### A komponensek szamossagabol a "pairwise-connectivity"
    summa = sum([len(c)*(len(c)-1)/2 if (len(c)>1) else 0 for c in lst])
    return summa

def h_component(lst): ### A komponensek szamossagabol a "pairwise-connectivity"
    return sum(lst)

def select_random(lst): ### Veletlen elem
    return random.choice(lst)

def best_nodes_CNP1A_Alg1(G,S): ### Legjobb pontok kigeneralasa
    selectedNodes = []
    min_pw = INF
    R = G.copy()
    R.remove_nodes_from(S)
    node_f_orig = f_pairwise(nx.connected_components(R))
    #print("-> S: ", S)
    #print("   node_f_orig = ", node_f_orig)
    for curr_node in S:
        R = G.copy()
        R.remove_nodes_from(list(set(S)-set([curr_node])))
        node_f = f_pairwise(nx.connected_components(R)) - node_f_orig
        #print("      node_f = ", node_f," (S: ",set(S)-set([curr_node]),")")

        if node_f < min_pw:
            selectedNodes.clear
            selectedNodes.append(curr_node)
            min_pw = node_f
        elif node_f == min_pw:
            selectedNodes.append(curr_node)

    return selectedNodes

# Vertex cover generalas: O(E log V)
# S = approx.min_weighted_vertex_cover(G)
# print("Vertex cover: ", S)
# print("A Vertex cover merete: ",len(S))

# CNP1a Alg1 G1
def CNP1a_1_G1(G):
    S = approx.min_weighted_vertex_cover(G)
    while len(S) > K:
        B = best_nodes_CNP1A_Alg1(G,S)
        #print("B: ", B)
        S = list(set(S)-set([select_random(B)]))
        #print(S)
    
    H = G.copy()
    H.remove_nodes_from(S)
    return [H,S]

def best_nodes_CNP1A_Alg2(G,S): ### Legjobb pontok kigeneralasa
    selectedNodes = []
    min_pw = INF
    P = G.copy()
    P.remove_nodes_from(S)
    node_f_orig = f_pairwise(nx.connected_components(P))
    SG = nx.nodes(G)
    #print("-> S: ", S)
    #print("   node_f_orig = ", node_f_orig)
    for curr_node in SG:
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

    return selectedNodes

# CNP1a Alg2 G2
def CNP1a_2_G1(G):
    S = []
    while len(S) < K:
        B = best_nodes_CNP1A_Alg2(G,S)
        #print("B: ", B)
        z = select_random(B)
        S.append(z)
        G.remove_nodes_from([z])
        #print(S)
    
    H = G.copy()
    H.remove_nodes_from(S)
    return [H,S]

# %%

def best_nodes_CNP3A_Alg2(G,S): ### Legjobb pontok kigeneralasa
    selectedNodes = []
    min_pw = INF
    P = G.copy()
    P.remove_nodes_from(S)
    node_f_orig = f_pairwise(nx.connected_components(P))
    SG = nx.nodes(G)
    #print("-> S: ", S)
    #print("   node_f_orig = ", node_f_orig)
    for curr_node in SG:
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

    return selectedNodes

# CNP1a Alg2 G2
def CNP3a_2_G1(G):
    S = []
    while len(S) < K:
        B = best_nodes_CNP3A_Alg2(G,S)
        #print("B: ", B)
        z = select_random(B)
        S.append(z)
        G.remove_nodes_from([z])
        #print(S)
    
    H = G.copy()
    H.remove_nodes_from(S)
    return [H,S]

#%%

def runCNPMethod(P, method):
    minVal = INF
    minR = P
    minSS = []
    for x in range(1,IterationCount):
        G = P.copy()
        [R,SS] = method(G)
        #print(SS)
        currVal = f_pairwise(nx.connected_components(R))
        #print(currVal)
        if currVal<minVal:
            minVal = currVal
            minR = R
            minSS = SS

    return [minVal, minR, minSS]

def RunAndPlotCNPMethod(G, method,plotNo):
    [minVal, minR, minSS] = runMethod(G,method)
    print(method.__name__)
    print(minSS)
    print(minVal)
    plt.subplot(plotNo)
    nx.draw(minR, with_labels=True) #, pos=nx.planar_layout(G), node_color='r', edge_color='b')

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

        #print("---> G after E: ",G.nodes())
        #print("---> G after N: ",G.edges())
        #print("---> S: ",S)
        #print("---> E: ",E)
        #print("-----> ",len(S)+len(E))
        #print("")
        #print(S)
    
    H = G.copy()
    H.remove_nodes_from(S)
    H.remove_edges_from(E)
    #print("G after E: ",H.nodes())
    #print("G after N: ",H.edges())
    
    return [H,S,E]

#%%

def runCNEPMethod(P, method):
    minVal = INF
    minR = P
    minSS = []
    minEE = []
    for x in range(1,IterationCount):
        G = P.copy()
        [R,SS, EE] = method(G)
        #print(SS)
        currVal = f_pairwise(nx.connected_components(R))
        #print(currVal)
        if currVal<minVal:
            minVal = currVal
            minR = R
            minSS = SS
            minEE = EE

    return [minVal, minR, minSS, minEE]

def RunAndPlotCNEPMethod(G, method,plotNo):
    [minVal, minR, minSS, minEE] = runCNEPMethod(G,method)
    print(method.__name__)
    print(minSS)
    print(minEE)
    print(minVal)
    plt.subplot(plotNo)
    nx.draw(minR, with_labels=True) #, pos=nx.planar_layout(G), node_color='r', edge_color='b')
#%%


#################
#plt.subplot(131)
#nx.draw(G, with_labels=True)
#################

#RunAndPlotCNPMethod(G,CNP1a_1_G1,131)
#RunAndPlotCNPMethod(G,CNP1a_2_G1,132)
#RunAndPlotCNPMethod(G,CNP3a_2_G1,133)

######## 
# RunAndPlotCNEPMethod(G,CNEP1a_2_G1,133)
########


n_pop = 10
n_iter = 10
r_crossover = 0.2
r_mutate = 0.3
r_retry = 0.5

print(genetic_algorithm(genObjectiveF,K1,K2,n_iter,n_pop,r_crossover,r_mutate,r_retry))


# %%