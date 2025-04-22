# compute all metrics for a test graph (with real target values)
# give in the list below the graph name that you want to compute its features, or all graphs in a directory
# computed features for the random subgraphs of each graph are saved in data/synthetic/graph_name.csv

datasets = ['Erdos_700_7500_grid_40_401_480.pickle']
#datasets = [f for f in os.listdir('Synthetic/') if 'Erdos' in f]


import os
import collections
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import ppi, Reddit
from torch_geometric.datasets import wikics, AttributedGraphDataset, Actor, GitHub, HeterophilousGraphDataset, Twitch, \
    Airports
from torch_geometric.datasets import Amazon, Coauthor, GNNBenchmarkDataset, NELL, CitationFull, Reddit, Reddit2, Flickr, \
    Yelp, AmazonProducts, Entities
import networkx as nx
from networkx.algorithms import community, bipartite
import csv
import datetime
import random
from sklearn.feature_selection import mutual_info_classif
import torch_geometric.transforms as T
from sklearn.ensemble import RandomForestClassifier
import util
from util import writeToReport
from sklearn.model_selection import cross_val_score
import pickle

def list_to_str(list):
    str_list = ''
    for l in list:
        str_list += str(l) + ','
    return str_list

def compute_features(dataset):
    start = datetime.datetime.now()
    graph_name = dataset
    print(graph_name)
    data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + dataset, 'rb')))
    print('data:')
    print(data)
    print(data.y)
    print(data.y[0])
    nodes_count = data.num_nodes
    edges_count = data.num_edges

    report_file_features = 'data/synthetic/' + graph_name + '.csv'
    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random, '+ list_to_str([ind for ind in range(0,73)]) +  ' Target ')

    G = nx.Graph()
    G = to_networkx(data, to_undirected=True)
    start_total = datetime.datetime.now()
    nodes_count = nx.number_of_nodes(G)
    print(nx.number_of_nodes(G))
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    H = G
    subgraph_nodes = list(G.nodes)
    print('Number of nodes: ')
    print(nx.number_of_nodes(H))
    print(nx.number_of_edges(H))
    features_count = 0
    # compute graphlet features
    '''
    for g in graphs:
        features_count += g.number_of_nodes()
    res = np.zeros(shape=(H.number_of_nodes(), features_count))
    feature_index = 0
    total_time_graphlets_list = []
    total_time_graphlets = datetime.timedelta()
    for graph_index, g in enumerate(graphs):
        print('Graphlet: ' + str(graph_index))
        start_time_graphlet = datetime.datetime.now()
        GM = nx.isomorphism.GraphMatcher(H, g)
        g_iter = GM.subgraph_isomorphisms_iter()
        for i, index_graphlet in enumerate(g_iter):
            keys = index_graphlet.keys()
            for key in keys:
                res[int(key)][int(index_graphlet[key])] = 1
        feature_index += g.number_of_nodes()
        end_time_graphlet = datetime.datetime.now()
        total_time_graphlet = datetime.timedelta()
        total_time_graphlet = (end_time_graphlet - start_time_graphlet)
        total_time_graphlets_list.append(total_time_graphlet)
        total_time_graphlets += total_time_graphlet
    res = np.delete(res,
                    [1, 4, 6, 7, 10, 11, 14, 15, 17, 18, 19, 22, 26, 27, 29, 30, 31, 35, 36, 41, 43, 44, 45, 49, 51,
                     53,
                     58,
                     61, 63, 64, 65, 66, 70, 75, 78, 80, 81,
                     85, 89, 90, 91, 95, 96, 99, 100, 101, 104, 105, 109, 111, 114, 116, 120, 121, 123, 124, 125,
                     129,
                     130,
                     131, 133, 134, 135, 136], axis=1)
    print('Graphlets time: ' + str(total_time_graphlets))
    '''
    start_time_features = datetime.datetime.now()

    start_time_degree_centrality = datetime.datetime.now()
    degree_centrality = nx.degree_centrality(H)
    end_time_degree_centrality = datetime.datetime.now()
    total_time_degree_centrality = datetime.timedelta()
    total_time_degree_centrality = (end_time_degree_centrality - start_time_degree_centrality)

    start_time_eigenvector_centrality = datetime.datetime.now()
    eigenvector_centrality = nx.eigenvector_centrality(H, max_iter=100, tol=1e-03)
    end_time_eigenvector_centrality = datetime.datetime.now()
    total_time_eigenvector_centrality = datetime.timedelta()
    total_time_eigenvector_centrality = (end_time_eigenvector_centrality - start_time_eigenvector_centrality)

    start_time_closeness_centrality = datetime.datetime.now()
    closeness_centrality = nx.closeness_centrality(H)
    end_time_closeness_centrality = datetime.datetime.now()
    total_time_closeness_centrality = datetime.timedelta()
    total_time_closeness_centrality = (end_time_closeness_centrality - start_time_closeness_centrality)

    start_time_harmonic_centrality = datetime.datetime.now()
    harmonic_centrality = nx.harmonic_centrality(H)
    end_time_harmonic_centrality = datetime.datetime.now()
    total_time_harmonic_centrality = datetime.timedelta()
    total_time_harmonic_centrality = (end_time_harmonic_centrality - start_time_harmonic_centrality)

    start_time_betweenness_centrality = datetime.datetime.now()
    betweenness_centrality = nx.betweenness_centrality(H)
    end_time_betweenness_centrality = datetime.datetime.now()
    total_time_betweenness_centrality = datetime.timedelta()
    total_time_betweenness_centrality = (end_time_betweenness_centrality - start_time_betweenness_centrality)

    start_time_coloring_lf = datetime.datetime.now()
    coloring_largest_first = nx.coloring.greedy_color(H, strategy='largest_first')
    end_time_coloring_lf = datetime.datetime.now()
    total_time_coloring_lf = datetime.timedelta()
    total_time_coloring_lf = (end_time_coloring_lf - start_time_coloring_lf)

    start_time_coloring_sl = datetime.datetime.now()
    coloring_smallest_last = nx.coloring.greedy_color(H, strategy='smallest_last')
    end_time_coloring_sl = datetime.datetime.now()
    total_time_coloring_sl = datetime.timedelta()
    total_time_coloring_sl = (end_time_coloring_sl - start_time_coloring_sl)

    start_time_coloring_is = datetime.datetime.now()
    coloring_independent_set = nx.coloring.greedy_color(H, strategy='independent_set')
    end_time_coloring_is = datetime.datetime.now()
    total_time_coloring_is = datetime.timedelta()
    total_time_coloring_is = (end_time_coloring_is - start_time_coloring_is)

    start_time_coloring_rs = datetime.datetime.now()
    coloring_random_sequential = nx.coloring.greedy_color(H, strategy='random_sequential')
    end_time_coloring_rs = datetime.datetime.now()
    total_time_coloring_rs = datetime.timedelta()
    total_time_coloring_rs = (end_time_coloring_rs - start_time_coloring_rs)

    start_time_coloring_dfs = datetime.datetime.now()
    coloring_connected_sequential_dfs = nx.coloring.greedy_color(H, strategy='connected_sequential_dfs')
    end_time_coloring_dfs = datetime.datetime.now()
    total_time_coloring_dfs = datetime.timedelta()
    total_time_coloring_dfs = (end_time_coloring_dfs - start_time_coloring_dfs)

    start_time_coloring_bfs = datetime.datetime.now()
    coloring_connected_sequential_bfs = nx.coloring.greedy_color(H, strategy='connected_sequential_bfs')
    end_time_coloring_bfs = datetime.datetime.now()
    total_time_coloring_bfs = datetime.timedelta()
    total_time_coloring_bfs = (end_time_coloring_bfs - start_time_coloring_bfs)

    start_time_node_clique_number = datetime.datetime.now()
    node_clique_number = nx.node_clique_number(H)
    end_time_node_clique_number = datetime.datetime.now()
    total_time_node_clique_number = datetime.timedelta()
    total_time_node_clique_number = (end_time_node_clique_number - start_time_node_clique_number)

    start_time_number_of_cliques = datetime.datetime.now()
    number_of_cliques = {n: sum(1 for c in nx.find_cliques(H) if n in c) for n in H}
    end_time_number_of_cliques = datetime.datetime.now()
    total_time_number_of_cliques = datetime.timedelta()
    total_time_number_of_cliques = (end_time_number_of_cliques - start_time_number_of_cliques)

    start_time_clustering_coefficient = datetime.datetime.now()
    clustering_coefficient = nx.clustering(H)
    end_time_clustering_coefficient = datetime.datetime.now()
    total_time_clustering_coefficient = datetime.timedelta()
    total_time_clustering_coefficient = (end_time_clustering_coefficient - start_time_clustering_coefficient)

    start_time_square_clustering = datetime.datetime.now()
    square_clustering = nx.square_clustering(H)
    end_time_square_clustering = datetime.datetime.now()
    total_time_square_clustering = datetime.timedelta()
    total_time_square_clustering = (end_time_square_clustering - start_time_square_clustering)

    start_time_average_neighbor_degree = datetime.datetime.now()
    average_neighbor_degree = nx.average_neighbor_degree(H)
    end_time_average_neighbor_degree = datetime.datetime.now()
    total_time_average_neighbor_degree = datetime.timedelta()
    total_time_average_neighbor_degree = (end_time_average_neighbor_degree - start_time_average_neighbor_degree)

    start_time_hubs = datetime.datetime.now()
    hubs, authorities = nx.hits(H)
    end_time_hubs = datetime.datetime.now()
    total_time_hubs = datetime.timedelta()
    total_time_hubs = (end_time_hubs - start_time_hubs)

    start_time_page_rank = datetime.datetime.now()
    page_rank = nx.pagerank(H)
    end_time_page_rank = datetime.datetime.now()
    total_time_page_rank = datetime.timedelta()
    total_time_page_rank = (end_time_page_rank - start_time_page_rank)

    start_time_core_number = datetime.datetime.now()
    core_number = nx.core_number(H)
    end_time_core_number = datetime.datetime.now()
    total_time_core_number = datetime.timedelta()
    total_time_core_number = (end_time_core_number - start_time_core_number)

    end_time_features = datetime.datetime.now()
    total_time_features = datetime.timedelta()
    total_time_features = (end_time_features - start_time_features)
    print('total time features: ')
    print(total_time_features)

    total_time_egonet = datetime.timedelta()
    total_time_triangles = datetime.timedelta()
    total_time_random = datetime.timedelta()

    X = np.zeros([nx.number_of_nodes(H), metrics_count])
    Y = np.zeros([nx.number_of_nodes(H)])
    for i, v in enumerate(H):

        start_time_degree = datetime.datetime.now()
        X[i][0] = H.degree(v)
        end_time_degree = datetime.datetime.now()
        total_time_degree = datetime.timedelta()
        total_time_degree = (end_time_degree - start_time_degree)
        X[i][1] = degree_centrality[subgraph_nodes[i]]
        neighborhood_degrees = [H.degree(n) for n in nx.neighbors(H, v)]
        if (len(neighborhood_degrees) == 0):
            max_neighbor_degree = 0
            min_neighbor_degree = 0
            std_neighbor_degree = 0
        else:
            max_neighbor_degree = np.max(neighborhood_degrees)
            min_neighbor_degree = np.min(neighborhood_degrees)
            std_neighbor_degree = np.std(neighborhood_degrees)
        X[i][2] = max_neighbor_degree
        X[i][3] = min_neighbor_degree
        X[i][4] = average_neighbor_degree[subgraph_nodes[i]]
        X[i][5] = std_neighbor_degree
        X[i][6] = eigenvector_centrality[subgraph_nodes[i]]
        X[i][7] = closeness_centrality[subgraph_nodes[i]]
        X[i][8] = harmonic_centrality[subgraph_nodes[i]]
        X[i][9] = betweenness_centrality[subgraph_nodes[i]]
        X[i][10] = coloring_largest_first[subgraph_nodes[i]]
        X[i][11] = coloring_smallest_last[subgraph_nodes[i]]
        X[i][12] = coloring_independent_set[subgraph_nodes[i]]
        X[i][13] = coloring_random_sequential[subgraph_nodes[i]]
        X[i][14] = coloring_connected_sequential_dfs[subgraph_nodes[i]]
        X[i][15] = coloring_connected_sequential_bfs[subgraph_nodes[i]]

        start_time_egonet = datetime.datetime.now()
        egonet = nx.ego_graph(G, v, radius=1)
        edges_within_egonet = nx.number_of_edges(egonet)
        end_time_egonet = datetime.datetime.now()
        total_time_egonet += (end_time_egonet - start_time_egonet)

        X[i][16] = edges_within_egonet
        X[i][17] = node_clique_number[subgraph_nodes[i]]
        X[i][18] = number_of_cliques[subgraph_nodes[i]]
        X[i][19] = clustering_coefficient[subgraph_nodes[i]]
        X[i][20] = square_clustering[subgraph_nodes[i]]
        X[i][21] = page_rank[subgraph_nodes[i]]
        X[i][22] = hubs[subgraph_nodes[i]]

        start_time_triangles = datetime.datetime.now()
        X[i][23] = nx.triangles(H, subgraph_nodes[i])
        end_time_triangles = datetime.datetime.now()
        total_time_triangles += (end_time_triangles - start_time_triangles)

        X[i][24] = core_number[subgraph_nodes[i]]

        start_time_random = datetime.datetime.now()
        X[i][25] = np.random.normal(0, 1, 1)[0]
        end_time_random = datetime.datetime.now()
        total_time_random += (end_time_random - start_time_random)

        Y[i] = data.y[subgraph_nodes[i]]
        print(Y[i])
        print(type(Y[i]))

    X = np.concatenate((X, res), axis=1)

    for i,x in enumerate(X):
        writeToReport(report_file_features, list_to_str(x) + str(Y[i]))
    writeToReport(report_file_features, '\n')

    end = datetime.datetime.now()
    total_time = datetime.timedelta()
    total_time = (end - start)
    print('Computing all faetures time: ' + str(total_time))

    report_file = 'reports/computing_time.csv'
    writeToReport(report_file,
                  'graph_name , nodes_count , edges_count, degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random,'  + ' Total Time ')
    writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(
        edges_count) + ',' + str(total_time_degree) + ',' + str(total_time_degree_centrality) + ',' + str(
        total_time_average_neighbor_degree) + ',' + str(total_time_average_neighbor_degree)
                  + ',' + str(total_time_average_neighbor_degree) + ',' + str(
        total_time_average_neighbor_degree) + ',' +
                  str(total_time_eigenvector_centrality) + ',' + str(total_time_closeness_centrality) + ',' + str(
        total_time_harmonic_centrality) + ',' + str(total_time_betweenness_centrality)
                  + ',' + str(total_time_coloring_lf) + ',' + str(total_time_coloring_sl) + ',' + str(
        total_time_coloring_is) + ',' + str(total_time_coloring_rs) + ',' + str(total_time_coloring_dfs)
                  + ',' + str(total_time_coloring_bfs) + ',' + str(total_time_egonet) + ',' + str(
        total_time_node_clique_number) + ',' + str(total_time_number_of_cliques) + ',' + str(
        total_time_clustering_coefficient)
                  + ',' + str(total_time_square_clustering) + ',' + str(total_time_page_rank) + ',' + str(
        total_time_hubs) + ',' + str(total_time_triangles) + ',' + str(total_time_core_number) + ',' + str(
        total_time_random) + ','  + str(total_time))


data_dir = "./datasets"
os.makedirs(data_dir, exist_ok=True)
datasets = ['Erdos_700_7500_grid_40_401_480.pickle']
for d in datasets:
    if (d == 'Cora' or d == 'CiteSeer' or d == 'PubMed'):
        dataset = Planetoid(root= data_dir, name=d)
    if (d == 'Photo' or d == 'Computers'):
        dataset = Amazon(root=data_dir, name=d)
    if (d == 'CS' or d == 'Physics'):
        dataset = Coauthor(root=data_dir, name=d)
    if (d == 'NELL'):
        dataset = NELL(root=data_dir)
    if (d == 'Cora_ML' or d == 'DBLP'):
        dataset = CitationFull(root=data_dir, name=d)
    if (d == 'Reddit'):
        dataset = Reddit(root=data_dir)
    if (d == 'Reddit2'):
        dataset = Reddit2(root=data_dir)
    if (d == 'Yelp'):
        dataset = Yelp(root=data_dir)
    if (d == 'AmazonProducts'):
        dataset = AmazonProducts(root=data_dir)
    if (d == 'AIFB' or d == 'AM' or d == 'MUTAG' or d == 'BGS'):
        dataset = Entities(root=data_dir, name=d)
    if (d == 'BlogCatalog' or d == 'Flickr' or d == 'Facebook' or d == 'Wiki'):
        dataset = AttributedGraphDataset(root=data_dir, name=d)
    if (d == 'Actor'):
        dataset = Actor(root=data_dir)
    if (d == 'GitHub'):
        dataset = GitHub(root=data_dir)
    if (d == 'Roman-empire' or d == 'Amazon-ratings' or d == 'Minesweeper' or d == 'Tolokers' or d == 'Questions'):
        dataset = HeterophilousGraphDataset(root=data_dir, name=d)
    if (d == 'DE' or d == 'EN' or d == 'ES' or d == 'FR' or d == 'PT' or d == 'RU'):
        dataset = Twitch(root=data_dir, name=d)
    if (d == 'USA' or d == 'Brazil' or d == 'Europe'):
        dataset = Airports(root=data_dir, name=d)

for dataset in datasets:
   compute_features(dataset)
