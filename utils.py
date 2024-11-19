import numpy as np
import torch
from torch.utils.data import Dataset
from Hyperparameter import *
from tqdm import tqdm

def scale_weights(weights):
    if len(weights) == 0:
        return weights

    log_weights = np.log(weights + 1)
    
    # log_min = np.min(log_weights)
    log_max = np.max(log_weights)

    # normalized_weights = (log_weights - log_min + epsilon) / (log_max - log_min + epsilon)
    normalized_weights = log_weights / log_max
    normalized_weights = torch.softmax(torch.tensor(normalized_weights, dtype=torch.float), dim=0).numpy()

    return normalized_weights

def select_neighbors(weight_out, weight_in, number=6):
    flag = False
    out_count, in_count = 0, 0
    for i in range(number):
        if len(weight_out) == 0 and len(weight_out):
            break

        if len(weight_out) == 0:
            flag = True
        elif len(weight_in) == 0:
            flag = False
        else:
            diff = out_count - in_count
            if diff > 0:
                if weight_out[0] > int(weight_in[0] * Multiplier_list[diff - 1 if diff < 4 else 2]):
                    flag = False
                else:
                    flag = True
            elif diff < 0:
                if weight_in[0] > int(weight_out[0] * Multiplier_list[-diff - 1 if -diff < 4 else 2]):
                    flag = True
                else:
                    flag = False
            else:
                if weight_in[0] > weight_out[0]:
                    flag = True
                else:
                    flag = False

        if flag == False:
            out_count += 1
            weight_out = weight_out[1:]
        else:
            in_count += 1
            weight_in = weight_in[1:]

    return out_count, in_count

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_global(seqs):
    len_data = [len(nowData) for nowData in seqs]
    N = max(len_data)
    new_seqs = [[0] + i + [0] * (N - len_data[index])  for index, i in enumerate(seqs)]
    return new_seqs


class Data(Dataset):
    def __init__(self, data, global_train):
        inputs, mask, max_len = handle_data(data[0])
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.global_train = handle_global(global_train)

    def __getitem__(self, index):
        u_input, mask, target, global_seqs = self.inputs[index], self.mask[index], self.targets[index], self.global_train[index]

        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items), 
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), 
                torch.tensor(global_seqs)]

    def __len__(self):
        return self.length


def get_global_seq_adj(global_graph, seqs):
    global_seq = []

    for seq in tqdm(seqs, colour='green', desc='Processing graph data', leave=False):
        seq = list(reversed(seq))
        u_input = np.array(seq)
        nodes = np.unique(u_input)
        final_nodes = nodes[:]

        for node in nodes:
            weight_out = np.array(global_graph[1][node])
            weight_in = np.array(global_graph[3][node])
            out_count, in_count = select_neighbors(weight_out, weight_in, number=number_neighbors)

            global_neighbors_out = np.array(global_graph[0][node][:out_count])
            diff_out = np.setdiff1d(global_neighbors_out, final_nodes)
            final_nodes = np.concatenate([final_nodes, diff_out], axis=0)

            global_neighbors_in = np.array(global_graph[2][node][:in_count])
            diff_in = np.setdiff1d(global_neighbors_in, final_nodes)
            final_nodes = np.concatenate([final_nodes, diff_in], axis=0)

        global_seq.append(list(final_nodes))

    return global_seq


def build_adj(global_graph, items, N):
    adj_array_out, adj_array_in = [], []
    for seq in items:
        adj_out = np.zeros((N, N), dtype=float)
        adj_in = np.zeros((N, N), dtype=float)
        indexs = np.nonzero(seq)[0]
        for index in indexs:
            node = seq[index]
            global_neighbors_out = np.array(global_graph[0][node])
            weight_out = np.array(global_graph[1][node])
            
            global_neighbors_in = np.array(global_graph[2][node])
            weight_in = np.array(global_graph[3][node])

            intersect_out = np.intersect1d(global_neighbors_out, seq)[:3]
            intersect_in = np.intersect1d(global_neighbors_in, seq)[:3]

            if node not in intersect_in and node not in intersect_out:
                adj_out[index, index] = self_weight
                adj_in[index, index] = self_weight

            inter_global_index_out = [np.where(global_neighbors_out == i)[0][0] for i in intersect_out]
            inter_global_index_in = [np.where(global_neighbors_in == i)[0][0] for i in intersect_in]
            inter_session_index_out = [np.where(seq == i)[0][0] for i in intersect_out]
            inter_session_index_in = [np.where(seq == i)[0][0] for i in intersect_in]

            weight_out = scale_weights(weight_out[inter_global_index_out])
            weight_in = scale_weights(weight_in[inter_global_index_in])
            adj_out[index, inter_session_index_out] = weight_out
            adj_in[inter_session_index_in, index] = weight_in

        adj_out, adj_in = torch.tensor(adj_out, dtype=torch.float), torch.tensor(adj_in, dtype=torch.float)
    
        adj_array_out.append(adj_out)
        adj_array_in.append(adj_in)

    return adj_array_out, adj_array_in


def get_N(items):

    nonzero_indices = torch.nonzero(items, as_tuple=False)
    last_nonzero_indices = torch.zeros(items.size(0), dtype=torch.long)
    for i in range(items.size(0)):
        row_indices = nonzero_indices[nonzero_indices[:, 0] == i, 1]
        if len(row_indices) > 0:
            last_nonzero_indices[i] = row_indices[-1]

    return max(last_nonzero_indices).item() + 1

def get_overlap(sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree