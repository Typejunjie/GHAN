import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output

class GlobalAggregator(nn.Module):
    def __init__(self, hidden_size, step=1, slope=0.2, dropout_rate=0.5) -> None:
        super(GlobalAggregator, self).__init__()

        self.step = step
        self.hidden_size = hidden_size
        self.slope = slope
        self.dropout_rate = dropout_rate

        self.w_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_4 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.a_out = nn.Linear(self.hidden_size, 1, bias=False)
        self.a_in = nn.Linear(self.hidden_size, 1, bias=False)

    def GNNCell(self, h_i, A_out, A_in):
        h_j_out = torch.matmul(A_out, h_i)
        h_j_in = torch.matmul(A_in, h_i)
        h_i, h_j_out, h_j_in = self.w_1(h_i), self.w_2(h_j_out), self.w_3(h_j_in)

        e_out = self.a_out(F.leaky_relu(h_i * h_j_out, negative_slope=self.slope))
        e_in = self.a_in(F.leaky_relu(h_i * h_j_in, negative_slope=self.slope))
        h_j_out = e_out * h_j_out
        h_j_in = e_in * h_j_in
        hidden = torch.cat((h_i, h_j_out, h_j_in), dim=-1)
        hidden = F.dropout(hidden, self.dropout_rate, training=self.training)
        hidden = torch.relu(self.w_4(hidden))
        return hidden

    def forward(self, hidden, A_in, A_out):
        for i in range(self.step):
            hidden = self.GNNCell(hidden, A_out, A_in)

        # h_array = torch.stack(h_array, dim=0)
        # hidden = torch.mean(h_array, dim=0).view(-1, self.hidden_size)
        return hidden
    

class LineConv(nn.Module):
    def __init__(self, layers, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.zeros(1, self.emb_size).to(item_embedding.device)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = torch.stack(seq_h, dim=0)
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        #session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        #session_emb_lgcn = torch.sum(session1, 0)
        session_emb_lgcn = torch.mean(torch.stack(session, dim=0), dim=0)
        return session_emb_lgcn