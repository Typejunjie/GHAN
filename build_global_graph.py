import pickle
from tqdm import tqdm
import numpy as np
import argparse
import os
from utils import *
from Hyperparameter import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str , default='diginetica', help='')
opt = parser.parse_args()

if opt.dataset == 'diginetica':
    n_node = 43097
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37483
elif opt.dataset == 'RetailRocket':
    n_node = 36968
elif opt.dataset == 'Tmall':
    n_node = 40727
elif opt.dataset == 'Nowplaying':
    n_node = 60416
else:
    raise KeyError

hop = 1

all_seq = pickle.load(open(f'./datasets/{opt.dataset}/all_train_seq.txt', 'rb'))

if not os.path.exists(f'./datasets/{opt.dataset}/unique_nodes.txt'):
    # unique item in train
    unique_nodes = []
    for seq in tqdm(all_seq, leave=False):
        for node in seq:
            if node not in unique_nodes:
                unique_nodes.append(node)
    pickle.dump(unique_nodes, open(f'./datasets/{opt.dataset}/unique_nodes.txt', 'wb'))
else:
    unique_nodes = pickle.load(open(f'./datasets/{opt.dataset}/unique_nodes.txt', 'rb'))

graph_node_out = {k: [] for k in unique_nodes}
graph_node_in = {k: [] for k in unique_nodes}

for seq in tqdm(all_seq):
    assert len(seq) > 0
    for i, node in enumerate(seq):
        if i != len(seq) - 1:
            # 出度
            # graph_node_out[node].append(seq[i + 1])
            graph_node_out[node].extend([seq[i + 1] for j in range(1)])
            if i + 2 < len(seq) and hop > 1:
                graph_node_out[node].extend([seq[i + 2] for j in range(1)])
            if i + 3 < len(seq) and hop > 2:
                graph_node_out[node].extend([seq[i + 3] for j in range(1)])
            # 入度
            # graph_node_in[seq[i + 1]].append(node)
            graph_node_in[seq[i + 1]].extend([node for j in range(1)])
            if i - 1 >= 0 and hop > 1:
                graph_node_in[node].extend([seq[i - 1] for j in range(1)])
            if i -2 >= 0 and hop > 2:
                graph_node_in[node].extend([seq[i - 2] for j in range(1)])

weight_out = {k: [] for k in unique_nodes}
weight_in = {k: [] for k in unique_nodes}

for key in graph_node_out:
    cache = np.array(graph_node_out[key])
    cache = np.sort(cache)
    seq = []
    cache_node = -1
    for i, node in enumerate(cache):
        if cache_node != node:
            cache_node = node
            seq.append(node)
            weight_out[key].append(1)
        else:
            weight_out[key][-1] += 1
    
    graph_node_out[key] = seq

for key in graph_node_in:
    cache = np.array(graph_node_in[key])
    cache = np.sort(cache)
    seq = []
    cache_node = -1
    for i, node in enumerate(cache):
        if cache_node != node:
            cache_node = node
            seq.append(node)
            weight_in[key].append(1)
        else:
            weight_in[key][-1] += 1
    
    graph_node_in[key] = seq

# 按权重从大到小排序
for key in graph_node_out:
    neighbor = np.array(graph_node_out[key])
    w = np.array(weight_out[key])
    sorted_index = np.argsort(-w)
    w = w[sorted_index]
    # 过滤权重小于1
    # w = w[w > 1]
    # neighbor = neighbor[sorted_index][:len(w)]
    neighbor = neighbor[sorted_index]
    graph_node_out[key] = neighbor.tolist()
    weight_out[key] = w.tolist()

for key in graph_node_in:
    neighbor = np.array(graph_node_in[key])
    w = np.array(weight_in[key])
    sorted_index = np.argsort(-w)
    w = w[sorted_index]
    # 过滤权重小于1
    # w = w[w > 1]
    # neighbor = neighbor[sorted_index][:len(w)]
    neighbor = neighbor[sorted_index]
    graph_node_in[key] = neighbor.tolist()
    weight_in[key] = w.tolist()

# 补全部分不在train序列中的item编号
for i in range(1, n_node + 2):
    if i not in graph_node_out:
        graph_node_out[i] = []
        weight_out[i] = []

for i in range(1, n_node + 2):
    if i not in graph_node_in:
        graph_node_in[i] = []
        weight_in[i] = []


# 构建全局图序列
train_seq = pickle.load(open(f'./datasets/{opt.dataset}/train.txt', 'rb'))[0]
test_seq = pickle.load(open(f'./datasets/{opt.dataset}/test.txt', 'rb'))[0]

global_train_seq = get_global_seq_adj((graph_node_out, weight_out, graph_node_in, weight_in), train_seq)
global_test_seq = get_global_seq_adj((graph_node_out, weight_out, graph_node_in, weight_in), test_seq)

pickle.dump(global_train_seq, open(f'./datasets/{opt.dataset}/global_train.txt', 'wb'))
pickle.dump(global_test_seq, open(f'./datasets/{opt.dataset}/global_test.txt', 'wb'))
pickle.dump((graph_node_out, weight_out, graph_node_in, weight_in), open(f'./datasets/{opt.dataset}/global_graph.txt', 'wb'))

