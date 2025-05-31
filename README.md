Code and datasets associated with the manuscript:
Learning to Rank Features to Enhance Graph Neural Networks for Graph Classification.
Authors: Fouad Alkhoury, Tamás Horváth, Christian Bauckhage, and Stefan Wrobel.

## Repository Structure

This folder contains the scripts used to run the experiments, we explain in turn each file: 

### generate_synthetic_graphs.py 
The script generates synthetic graphs using the following parameters: 
base graph: Erdos or Barabasi, nodes_count in the base graph, edges count in Erdos, or m0 in Barabasi, 
count of motifs, motifs shape: path, house, grid, star, cycle, noisy edges. 
Generated graphs are saved under Synthetic/

### generate_features_ranking.py 
It generates feature ranking based on their importance in the random subgraphs. 
To run the script, include datasets to generate feature ranking for them. 
The generated ranking (resp. importance) is saved fo each graph in data/synthetic/ranking_synthetic (resp. importance_synthetic)

### compute_features.py 
The script computes all features for a test graph (with real target values). 
To run the script, give in the list the graph name that you want to compute its features, or all graphs in a directory. 
The computed features of each graph are saved in data/synthetic/graph_name.csv

### feature_learning_gnn.py 
It implements the FR-GNN model, it is the feature learning graph classification model that learns the ranking of graphs. 
Names of training graphs are stored in the file synthetic.train, the learned model is saved under model/

### test_model.py 
The script predicts for a query graph G the ranking of its features. 
Query graphs' (test graphs') name are stored in data/synthetic.train .
The predicted ranking and importance are written in reports/rank_predicted and reports/importance_predicted

### train_nc_gnn.py 
The script trains a node classification gnn model NC-GNN for a query graph g after splitting nodes into training, validation, and testing. 
The feature vector for each node consists of the first k features according to the predicted ranking from reports/rank_predicted.csv 
Results are saved in k_results 

The folder 'Synthetic' contains some generated synthetic graphs, we omit uploading more than 3000 synthetic graphs due to the large space needed. 
The results, reports, times are exported to 'reports' folder. 



