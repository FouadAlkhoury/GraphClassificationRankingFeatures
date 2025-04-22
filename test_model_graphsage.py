import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import ReadData
import os
import pickle
import torch_geometric
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
import util
from util import writeToReport
from torch_geometric.loader import DataLoader
from collections import OrderedDict
import datetime

features_count = 26

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(features_count, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(64 ,features_count)

    def forward(self, x, edge_index ,batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x ,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        out1 = self.lin1(x)

        return out1


def predict(data):

    model.eval()
    out1 = model(data.x, data.edge_index, data.batch)

    return out1


X_train = []
Y_train = []
X_test= []
Y_test = []
dataPath='data'
dataset_length = 48
train_length = int(dataset_length * (1) / (48))
X = ReadData.readData('test','test',dataPath)
print(X)
#print(Y)
offset = 0
#Y = Y[offset:offset+dataset_length]
X = X[offset:offset+dataset_length]
#Y = np.reshape(Y,(len(Y),features_count,1))
batch = 1
'''
y_all = []
y_all_test = []

for y in Y:
    tmp = []
    for f in y:
        tmp.append(f)
    y_all.append(tmp)
'''

def add_attributes(dataset):
    data_list = []
    for i, data in enumerate(dataset):

        #data.y = y_all[i]
        x_train = np.ones((data.num_nodes,features_count),dtype=np.float32)
        x_train = np.array(x_train)
        x_train = torch.from_numpy(x_train)
        data.x = x_train
        data_list.append(data)

    return data_list

model = GraphSAGE(64)
#model.load_state_dict(torch.load('Synthetic/model.pth'))
model=torch.load('model/model_graphsage.pth')#

X_train = X[:train_length]
#Y_train = Y[:train_length]

#Y_test = Y[train_length:dataset_length]
X_test = X[train_length:]
#Y_test = Y[train_length:]

dataset_list_test = []
graphs_list = [torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + x,'rb'))) for x in X]
graphs_list = graphs_list[:dataset_length]
dataset_list = add_attributes(graphs_list)
dataset_list_test = dataset_list[train_length:]

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

report_file_features = 'Synthetic/' + 'test_counting.csv'

writeToReport(report_file_features, 'Test Graph, Similaritiy, Ranking ')

k = 5
similarity = 0.0

test_loader = DataLoader(dataset_list_test, batch_size=1, shuffle=False)


def list_to_str(list):
    str_list = ''
    for l in list:
        #print(l)
        str_list += str(l) + ','
    return str_list


y_pred_list = []
y_pred_rank_list = []
y_test_list = []


time = datetime.datetime.now()
ranking_prediction_file = 'reports/rank_pred_graphsage_'+str(time)+'.csv'
ranking_test_file = 'reports/rank_test_graphsage_'+str(time)+'.csv'
importance_prediction_file = 'reports/importance_pred_graphsage_'+str(time)+'.csv'
testing_time_file = 'reports/testing_time.csv'

writeToReport(ranking_prediction_file,'graph, top 5 similarity, ranking')
writeToReport(ranking_test_file,'graph, ranking')
writeToReport(importance_prediction_file,'graph, importance')
writeToReport(testing_time_file,'graph, time')


for i,data in enumerate(test_loader):
    test_time_start = datetime.datetime.now()
    output_list = predict(data).tolist()[0]
    test_time_end = datetime.datetime.now()
    test_time = datetime.timedelta()
    test_time = (test_time_end - test_time_start)
    print(test_time)
    writeToReport(testing_time_file,X_test[i]+','+str(test_time))
    print('output list: ')
    print(output_list)

    y_pred = sorted(range(len(output_list)), key= lambda k: output_list[k],reverse=True)

    print('Y_pred: ' + str(output_list))
    print('Y_pred_rank: ' + str(y_pred))
    y_pred_list.append(output_list)
    y_pred_rank_list.append(y_pred)
    #y_test = [y[0] for y in Y_test[i]]
    #y_test_rank = sorted(range(len(output_list)), key=lambda k: y_test[k], reverse=True)
    #print('Y_test' + str(y_test))
    #print('Y_test_rank' + str(y_test_rank))
    #y_test_list.append(y_test_rank)

    y_pred_top_k = y_pred[:k]
    #y_test_top_k = y_test_rank[:k]
    #jaccard_sim = jaccard(y_pred_top_k,y_test_top_k)
    jaccard_sim = 0
    #print(jaccard_sim)
    #similarity += jaccard_sim
    writeToReport(report_file_features, X_test[i] +','+ str(jaccard_sim) + ',' + list_to_str(y_pred))

    writeToReport(ranking_prediction_file,X_test[i] + ',' + str(jaccard_sim) + ',' + list_to_str(y_pred))
    #writeToReport(ranking_test_file, X_test[i] + ',' + list_to_str(y_test_rank))
    writeToReport(importance_prediction_file, X_test[i] + ',' + list_to_str(output_list))



#print('avg sim: ')
#print(similarity / len(Y_test))

