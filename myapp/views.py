from django.shortcuts import render,HttpResponse
import copy
import networkx as nx
import numpy as np
import random
import csv
import pandas as pd
import scipy as sp
import matplotlib

import matplotlib.pyplot as plt
# Create your views here.

def index(req):
    return render(req,'index.html')

def centrality(req):
    return render(req,'centrality.html')

def linearThr(req):
    return render(req,'linearThr.html')

def icThr(req):
    return render(req,'icThr.html')

def aboutProj(req):
    return render(req,'aboutProj.html')

def showdata(req):
    # fileinp="C:\Users\DELL\Desktop\unweighted_graph.csv"
    fileinp="vla"
    fileinp = req.FILES['file']
    df = pd.read_csv(fileinp)

# Convert the dataframe to a list of tuples
    edges = [tuple(x) for x in df.to_numpy()]

# Create a directed graph from the edgelist
    G = nx.DiGraph()
    G.add_edges_from(edges)
    model=req.POST.get('model')
    centrality=req.POST.get('centrality')
  
    # G = nx.gnp_random_graph(10, 0.2)
    # G = nx.read_edgelist(fileinp, delimiter=',')
    # a= np.random.randint(0, 9, size=40)

    def linear_threshold_model(G, seeds,steps):   

       
  # make sure the seeds are in the graph
        for s in seeds:
            if s not in G.nodes():
                raise Exception("seed", s, "is not in graph")

  # change to directed graph
        if not G.is_directed():
            DG = G.to_directed()
        else:
            DG = copy.deepcopy(G)

  # init thresholds
        for n in DG.nodes():
            if 'threshold' not in DG._node[n]:
                DG._node[n]['threshold'] = 0.5
            elif DG._node[n]['threshold'] > 1:
                raise Exception("node threshold:", DG._node[n]['threshold'], "cannot be larger than 1")

  # init influences
        in_deg = DG.in_degree()
        for e in DG.edges():
            if 'influence' not in DG[e[0]][e[1]]:
                DG[e[0]][e[1]]['influence'] = 1.0 / in_deg[e[1]]
            elif DG[e[0]][e[1]]['influence'] > 1:
                raise Exception("edge influence:", DG[e[0]][e[1]]['influence'], "cannot be larger than 1")

        # perform diffusion
        A = copy.deepcopy(seeds)
        if steps <= 0:
            # perform diffusion until no more nodes can be activated
            return _diffuse_all(DG, A)
        # perform diffusion for at most "steps" rounds only
        return _diffuse_k_rounds(DG, A, steps)

    def _diffuse_all(G, A):
        layer_i_nodes = [ ]
        layer_i_nodes.append([i for i in A])
        while True:
            len_old = len(A)
            A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
            layer_i_nodes.append(activated_nodes_of_this_round)
            if len(A) == len_old:
                break
        return layer_i_nodes

    def _diffuse_k_rounds(G, A, steps):
        layer_i_nodes = [ ]
        layer_i_nodes.append([i for i in A])
        while steps > 0 and len(A) < len(G):
            len_old = len(A)
            A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
            layer_i_nodes.append(activated_nodes_of_this_round)
            if len(A) == len_old:
                break
            steps -= 1
        return layer_i_nodes

    def _diffuse_one_round(G, A):
        activated_nodes_of_this_round = set()
        for s in A:
            nbs = G.successors(s)
            for nb in nbs:
                if nb in A:
                    continue
                active_nb = list(set(G.predecessors(nb)).intersection(set(A)))
                if _influence_sum(G, active_nb, nb) >= G._node[nb]['threshold']:
                    activated_nodes_of_this_round.add(nb)
        A.extend(list(activated_nodes_of_this_round))
        return A, list(activated_nodes_of_this_round)

    def _influence_sum(G, froms, to):
        influence_sum = 0.0
        for f in froms:
            influence_sum += G[f][to]['influence']
        return influence_sum
    
    

    
    def independent_cascade(G, seeds,steps):   
    
        for s in seeds:
            if s not in G.nodes():
                raise Exception("seed", s, "is not in graph")

    # change to directed graph
        if not G.is_directed():
            DG = G.to_directed()
        else:
            DG = copy.deepcopy(G)

    # init activation probabilities
        for e in DG.edges():
            if 'act_prob' not in DG[e[0]][e[1]]:
                DG[e[0]][e[1]]['act_prob'] = 0.1
            elif DG[e[0]][e[1]]['act_prob'] > 1:
                raise Exception("edge activation probability:", \
                DG[e[0]][e[1]]['act_prob'], "cannot be larger than 1")

    # perform diffusion
        A = copy.deepcopy(seeds)  # prevent side effect
        if steps <= 0:
        # perform diffusion until no more nodes can be activated
            return _diffuse_all_ic(DG, A)
    # perform diffusion for at most "steps" rounds
        return _diffuse_k_rounds_ic(DG, A, steps)

    def _diffuse_all_ic(G, A):
        tried_edges = set()
        layer_i_nodes = [ ]
        layer_i_nodes.append([i for i in A])  # prevent side effect
        while True:
            len_old = len(A)
            (A, activated_nodes_of_this_round, cur_tried_edges) = \
                _diffuse_one_round_ic(G, A, tried_edges)
            layer_i_nodes.append(activated_nodes_of_this_round)
            tried_edges = tried_edges.union(cur_tried_edges)
            if len(A) == len_old:
                break
        return layer_i_nodes

    def _diffuse_k_rounds_ic(G, A, steps):
        tried_edges = set()
        layer_i_nodes = [ ]
        layer_i_nodes.append([i for i in A])
        while steps > 0 and len(A) < len(G):
            len_old = len(A)
            (A, activated_nodes_of_this_round, cur_tried_edges) = \
                _diffuse_one_round_ic(G, A, tried_edges)
            layer_i_nodes.append(activated_nodes_of_this_round)
            tried_edges = tried_edges.union(cur_tried_edges)
            if len(A) == len_old:
                break
            steps -= 1
        return layer_i_nodes

    def _diffuse_one_round_ic(G, A, tried_edges):
        activated_nodes_of_this_round = set()
        cur_tried_edges = set()
        for s in A:
            for nb in G.successors(s):
                if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
                    continue
                if _prop_success(G, s, nb):
                    activated_nodes_of_this_round.add(nb)
                cur_tried_edges.add((s, nb))
        activated_nodes_of_this_round = list(activated_nodes_of_this_round)
        A.extend(activated_nodes_of_this_round)
        return A, activated_nodes_of_this_round, cur_tried_edges

    def _prop_success(G, src, dest):
        return random.random() <= G[src][dest]['act_prob']




    k = 3
    seed_nodes=set()
    if centrality=='Degree':
        degree_centrality = nx.degree_centrality(G)        
        seed_nodes = list(sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:k])
    if centrality=='Closeness':
        closeness_centrality = nx.closeness_centrality(G)
        seed_nodes = list(sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[:k])

    if centrality=='Betweeness':  
        betweenness_centrality = nx.betweenness_centrality(G) 
        seed_nodes = list(sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:k])

    if centrality=='Page Rank':
        pagerank = nx.pagerank(G)
        seed_nodes = list(sorted(pagerank, key=pagerank.get, reverse=True)[:k])

    if centrality=='Eigen Vector':
        eigenvector_centrality = nx.eigenvector_centrality(G)   
        seed_nodes = list(sorted(eigenvector_centrality, key=eigenvector_centrality.get, reverse=True)[:k])

   

    nx.draw(G, with_labels=True)
    plt.show()
    
    final=[]
    node_colors1={}
    
    for i in range(len(seed_nodes)):

            node_colors1[seed_nodes[i]]='grey'
          
            i=i+1

    nx.draw(G, with_labels=True,node_color=[node_colors1.get(i, 'blue') for i in G.nodes()])
    plt.show()
    if model=='Linear Threshold Model':
        final=linear_threshold_model(G,seed_nodes,0)
    if model=='Independent Cascade':
        final=independent_cascade(G,seed_nodes,0)
    node_colors={}
    
    for i in range(len(final)):
        for j in range(len(final[i])):
            node_colors[final[i][j]]='red'
            j=j+1
        
        i=i+1

    nx.draw(G, with_labels=True,node_color=[node_colors.get(i, 'blue') for i in G.nodes()])
    plt.show()
    return render(req,"index.html",{'nodes':G.nodes(),'seedNodes':seed_nodes,'output':final,'centrality':centrality,'Model':model})   






