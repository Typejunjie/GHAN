import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator, LineConv
from torch.nn import Module
import torch.nn.functional as F
import pickle
from utils import *


class CombineGraph(Module):
    def __init__(self, opt, num_node):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.beta = opt.beta
        self.global_graph = pickle.load(open(f'./datasets/{opt.dataset}/global_graph.txt', 'rb'))

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.GlobalAggregator_1 = GlobalAggregator(self.dim, dropout_rate=0.5)
        self.GlobalAggregator_2 = GlobalAggregator(self.dim, dropout_rate=0.5)
        self.GlobalAggregator_3 = GlobalAggregator(self.dim, dropout_rate=0.5)
        # self.LineConv = LineConv(layers=1, emb_size=self.dim)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.a = nn.Parameter(torch.Tensor(1))
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        return select

    def forward(self, inputs, adj, mask_item, items, items_g, A_out, A_in):
        N = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        h_g = self.embedding(items_g)
        h_array = [h_g]
        h_array.append(self.GlobalAggregator_1(h_array[-1], A_out, A_in))
        h_array.append(self.GlobalAggregator_2(h_array[-1], A_out, A_in))
        h_array.append(self.GlobalAggregator_3(h_array[-1], A_out, A_in))
        h_g = torch.mean(torch.stack(h_array, dim=0), dim=0)
        # LineConv
        # h_lgcn = self.LineConv(self.embedding.weight, D, A, inputs, N)

        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_g = F.dropout(h_g, self.dropout_global, training=self.training)[:, :N]
        output = self.a * h_local + h_g * (1 - self.a)
        return output
    
    # def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
    #     def row_column_shuffle(embedding):
    #         corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
    #         corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
    #         return corrupted_embedding
    #     def score(x1, x2):
    #         return torch.sum(torch.mul(x1, x2), 1)

    #     pos = score(sess_emb_hgnn, sess_emb_lgcn)
    #     neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    #     one = torch.ones(neg1.shape[0], dtype=torch.float).to(sess_emb_hgnn.device)
    #     # one = zeros = torch.ones(neg1.shape[0])
    #     con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    #     return con_loss * self.beta
    

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    # return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs, items_g = data 

    N = get_N(items)
    N_g = get_N(items_g)
    
    alias_inputs = trans_to_cuda(alias_inputs).long()[:,:N]
    items = trans_to_cuda(items).long()[:,:N]
    items_g = trans_to_cuda(items_g).long()[:, :N_g]
    adj = trans_to_cuda(adj).float()[:,:N, :N]
    mask = trans_to_cuda(mask).long()[:,:N]
    inputs = trans_to_cuda(inputs).long()[:,:N]

    # A_hat, D_hat = get_overlap(items.cpu().numpy())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))

    adj_out, adj_in = build_adj(model.global_graph, items_g.cpu().numpy(), N_g)
    adj_out, adj_in = torch.stack(adj_out, dim=0), torch.stack(adj_in, dim=0)

    adj_out = trans_to_cuda(adj_out).float()
    adj_in = trans_to_cuda(adj_in).float()

    # hidden, h_lgcn = model(items, adj, mask, inputs, items_g, adj_out, adj_in, A_hat, D_hat)
    hidden = model(items, adj, mask, inputs, items_g, adj_out, adj_in)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    select = model.compute_scores(seq_hidden, mask)
    # con_loss = model.SSL(select, h_lgcn)

    b = model.embedding.weight[1:]  # n_nodes x latent_size
    scores = torch.matmul(select, b.transpose(1, 0))
    return targets, scores


def train_test(model, train_data, test_data, writer, epoch, topk):
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    length = len(train_loader)
    for i, data in enumerate(tqdm(train_loader, colour='green', desc=f'Epoch {epoch}', leave=False)):
        model.optimizer.zero_grad()
        # targets, scores, con_loss = forward(model, data)
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        # loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

        if i % 10 == 0:
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * length + i)
            
    tqdm.write('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    writer.add_scalar('loss/train_loss', total_loss, epoch)

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    hit, mrr = [[] for i in topk], [[] for i in topk]
    total_loss = 0.0

    for data in tqdm(test_loader, colour='green', desc='Estimating', leave=False):
        # targets, scores, con_loss = forward(model, data)
        targets, scores = forward(model, data)
        loss = model.loss_function(scores, trans_to_cuda(targets).long() - 1)
        # loss = loss + con_loss
        total_loss += loss.item()
        targets = targets.numpy()

        for index, i in enumerate(topk):
            sub_scores = scores.topk(i)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit[index].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr[index].append(0)
                else:
                    mrr[index].append(1 / (np.where(score == target - 1)[0][0] + 1))

    for index, j in enumerate(topk):
        hit[index] = np.mean(hit[index]) * 100
        mrr[index] = np.mean(mrr[index]) * 100
        writer.add_scalar(f'index/hit@{j}', hit[index], epoch)
        writer.add_scalar(f'index/mrr@{j}', mrr[index], epoch)

    writer.add_scalar('loss/test_loss', total_loss, epoch)
    return hit, mrr
