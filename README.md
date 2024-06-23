Code and data accompanying the paper

Alkhoury et al.: 
Improving Graph Neural Networks through Feature Learning.

## Repository Structure

The file generate_synthetic_graphs.py generates synthetic graphs using the following parameters:
base graph: Erdos or Barabasi, nodes_count in the base graph, edges count in Erdos, or m0 in Barabasi,
count of motifs, motifs shape: path, house, grid, star, cycle, noisy edges.
generated graphs are saved under Synthetic/

File generate_features_ranking.py  generate feature ranking based on their importance in the random subgraphs.
To run the script, include datasets to generate feature ranking for them
The generated ranking (resp. importance) is saved fo each graph in data/synthetic/ranking_synthetic (resp. importance_synthetic)

File compute_features.py computes all metrics for a test graph (with real target values)
To run the script, give in the list the graph name that you want to compute its features, or all graphs in a directory
The computed features for the random subgraphs of each graph are saved in data/synthetic/graph_name.csv

File feature_learning.py contains the feature learning graph classification model
Names of training graphs are stored in the file synthetic.train, the learned model is saved under model/

File test_model predicts for a query graph G the ranking of its features.
Query graphs names are stored in synthetic.train
The predicted ranking and importance are written in reports/rank_predicted and reports/importance_predicted

train_nc_gnn trains a node classification gnn model NC-GNN for a query graph g after splitting nodes into training, validation, and testing.
The feature vector for each node consists of the first k fetures according to the predicted ranking from reports/rank_predicted.csv
Results are saved in k_results

The folder 'Synthetic' contains some generated synthetic graphs.
The results, reports, times are exported to 'reports' folder.



