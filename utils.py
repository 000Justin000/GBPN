import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import cnetworkx as nx
import numba
import torch_sparse
from torch_sparse import SparseTensor, coalesce
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, subgraph, remove_self_loops, to_undirected, is_undirected, contains_self_loops, stochastic_blockmodel_graph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
from datetime import datetime, timedelta
from collections import defaultdict


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.ReLU(), dropout_p=0.0):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(self.dropout(x)))

        return self.linears[-1](self.dropout(x))


class GMLP(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.ReLU(), dropout_p=0.0):
        super(GMLP, self).__init__()
        self.mlp = MLP(dim_in, dim_out, dim_hidden, num_hidden, activation, dropout_p)

    def forward(self, x, edge_index, **kwargs):
        return F.log_softmax(self.mlp(x), dim=-1)


class SGC(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, dropout_p=0.0):
        super(SGC, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden)
        self.conv2 = GCNConv(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, activation=nn.ReLU(), dropout_p=0.0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden)
        self.conv2 = GCNConv(dim_hidden, dim_out)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class SAGE(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, activation=nn.ReLU(), dropout_p=0.0):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(dim_in, dim_hidden)
        self.conv2 = SAGEConv(dim_hidden, dim_out)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GAT(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, activation=nn.ELU(), dropout_p=0.0, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_in, dim_hidden, heads=num_heads, dropout=dropout_p)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(dim_hidden*num_heads, dim_out, heads=num_heads, concat=False, dropout=dropout_p)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class SumConv(MessagePassing):
    def __init__(self):
        super(SumConv, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class BPConv(MessagePassing):
    def __init__(self, n_channels, learn_H=False):
        super(BPConv, self).__init__(aggr='add')
        self.learn_H = learn_H
        dim_param = n_channels * (n_channels + 1) // 2
        self.param = nn.Parameter(torch.zeros(dim_param))
        self.n_channels = n_channels

    def get_logH(self):
        logT = torch.zeros(self.n_channels, self.n_channels).to(self.param.device)
        rid, cid = torch.tril_indices(self.n_channels, self.n_channels, 0)
        logT[rid, cid] = F.logsigmoid(self.param * 10.0)
        logH = (logT + logT.transpose(0, 1).triu(1))
        return (logH if self.learn_H else logH.detach().fill_diagonal_(0.0))

    def forward(self, x, edge_index, edge_weight, info):
        # x has shape [N, n_channels]
        # edge_index has shape [2, E]
        # info has 3 fields: 'log_b0', 'log_msg_', 'rv'
        return self.propagate(edge_index, edge_weight=edge_weight, x=x, info=info)

    def message(self, x_j, edge_weight, info):
        # x_j has shape [E, n_channels]
        logC = edge_weight.unsqueeze(-1).unsqueeze(-1) * self.get_logH().unsqueeze(0)
        log_msg_raw = torch.logsumexp((x_j - info['log_msg_'][info['rv']]).unsqueeze(-1) + logC, dim=-2)
        log_msg = log_normalize(log_msg_raw)
        info['log_msg_'] = log_msg
        return log_msg

    def update(self, agg_log_msg, info):
        log_b_raw = info['log_b0'] + agg_log_msg + (info['agg_scaling']-1.0).unsqueeze(-1) * agg_log_msg.detach()
        log_b = log_normalize(log_b_raw)
        return log_b


class BPGNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=0, activation=nn.ReLU(), dropout_p=0.0, learn_H=False):
        super(BPGNN, self).__init__()
        self.transform_ego = nn.Sequential(MLP(dim_in, dim_out, dim_hidden, num_hidden, activation, dropout_p), nn.LogSoftmax(dim=-1))
        self.bp_conv = BPConv(dim_out, learn_H)

    def forward(self, x, edge_index, edge_weight=None, agg_scaling=None, rv=None, phi=None, K=5):
        log_b0 = self.transform_ego(x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        if agg_scaling is None:
            agg_scaling = torch.ones(x.shape[0]).to(x.device)
        if rv is None:
            edge_index, edge_weight, rv = process_edge_index(x.shape[0], edge_index, edge_weight)
        if phi is not None:
            log_b0 = log_b0 + phi * agg_scaling.unsqueeze(-1)
        num_classes = log_b0.shape[-1]
        info = {'log_b0': log_b0,
                'log_msg_': (-np.log(num_classes)) * torch.ones(edge_index.shape[1], num_classes).to(x.device),
                'rv': rv,
                'agg_scaling': agg_scaling}
        log_b = log_b0
        for _ in range(K):
            log_b = self.bp_conv(log_b, edge_index, edge_weight, info)
        return log_b


class SubgraphSampler:

    def __init__(self, num_nodes, x, y, edge_index, edge_weight=None):
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = None if (edge_weight is None) else edge_weight
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)).t()
        self.deg = degree(edge_index[1], num_nodes)

    def get_generator(self, mask, batch_size, num_hops, size, device):
        idx = mask.nonzero(as_tuple=True)[0]
        n_batch = math.ceil(idx.shape[0] / batch_size)

        def generator():
            for batch_nodes in idx[torch.randperm(idx.shape[0])].chunk(n_batch):
                batch_size = batch_nodes.shape[0]

                # create subgraph from neighborhood
                subgraph_nodes = batch_nodes
                for _ in range(num_hops):
                    _, subgraph_nodes = self.adj_t.sample_adj(subgraph_nodes, size, replace=False)
                subgraph_size = subgraph_nodes.shape[0]

                assert torch.all(subgraph_nodes[:batch_size] == batch_nodes)
                subgraph_edge_index, subgraph_edge_weight = subgraph(subgraph_nodes, self.edge_index, self.edge_weight, relabel_nodes=True)
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_weight)

                yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                      subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                      subgraph_edge_index.to(device), None if (subgraph_edge_weight is None) else subgraph_edge_weight.to(device), subgraph_rv.to(device)

        return generator()


class SubtreeSampler:

    def __init__(self, num_nodes, x, y, edge_index, edge_weight=None):
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.G = nx.empty_graph(num_nodes)
        if edge_weight is None:
            self.G.add_edges_from(list(zip(edge_index[0].tolist(), edge_index[1].tolist())))
        else:
            self.G.add_weighted_edges_from(list(zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist())))
        self.deg = degree(edge_index[1], num_nodes)

    def get_generator(self, mask, batch_size, num_hops, size, device):
        idx = mask.nonzero(as_tuple=True)[0]
        n_batch = math.ceil(idx.shape[0] / batch_size)

        def generator():
            for batch_nodes in idx[torch.randperm(idx.shape[0])].chunk(n_batch):
                batch_size = batch_nodes.shape[0]

                T = nx.empty_graph(batch_size)
                subgraph_nodes = batch_nodes.tolist()

                def dfs(r, rid):
                    stack = [(r, rid, 0, -1)] if num_hops > 0 else []
                    while len(stack) > 0:
                        u, uid, d, p = stack.pop()
                        nbrs = set(self.G.neighbors(u)) - set([p])
                        for v in random.sample(nbrs, min(len(nbrs), size)):
                            subgraph_nodes.append(v)
                            vid = T.number_of_nodes()
                            T.add_node(vid)
                            if self.edge_weight is None:
                                T.add_edge(uid, vid)
                            else:
                                T.add_edge(uid, vid, weight=self.G[u][v]['weight'])
                            if num_hops > d+1:
                                stack.append((v, vid, d+1, u))

                for (rid, r) in enumerate(batch_nodes.tolist()):
                    dfs(r, rid)

                subgraph_nodes = torch.tensor(subgraph_nodes, dtype=torch.int64)
                subgraph_size = subgraph_nodes.shape[0]
                if self.edge_weight is None:
                    T_edge_index = torch.tensor(list(T.edges), dtype=torch.int64).t() if len(T.edges) == 0 else torch.zeros(2, 0, dtype=torch.int64) 
                    T_edge_weight = None
                else:
                    T_ew = nx.get_edge_attributes(T, 'weight')
                    T_edge_index = torch.tensor(list(T_ew.keys()), dtype=torch.int64).t() if len(T_ew.keys()) == 0 else torch.zeros(2, 0, dtype=torch.int64) 
                    T_edge_weight = torch.tensor(list(T_ew.values()))
                subgraph_edge_index, subgraph_edge_weight = T_edge_index, T_edge_weight
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_weight)

                yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                      subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                      subgraph_edge_index.to(device), None if (subgraph_edge_weight is None) else subgraph_edge_weight.to(device), subgraph_rv.to(device)

        return generator()


class CSubtreeSampler:

    def __init__(self, num_nodes, x, y, edge_index, edge_weight=None):
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.G = nx.empty_graph(num_nodes)
        if edge_weight is None:
            self.G.add_edges_from(list(zip(edge_index[0].tolist(), edge_index[1].tolist())))
        else:
            self.G.add_weighted_edges_from(list(zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist())))
        self.deg = degree(edge_index[1], num_nodes)

    def get_generator(self, mask, batch_size, num_hops, size, device):
        idx = mask.nonzero(as_tuple=True)[0]
        n_batch = math.ceil(idx.shape[0] / batch_size)

        def generator():
            for batch_nodes in idx[torch.randperm(idx.shape[0])].chunk(n_batch):
                batch_size = batch_nodes.shape[0]

                T = nx.sample_subtree(self.G, batch_nodes.tolist(), num_hops, size)
                subgraph_nodes = torch.tensor(T.get_nodes(), dtype=torch.int64)
                subgraph_size = subgraph_nodes.shape[0]
                if self.edge_weight is None:
                    T_edge_index = torch.tensor(T.get_edges(), dtype=torch.int64).t() if len(T.get_edges()) > 0 else torch.zeros(2, 0, dtype=torch.int64)
                    T_edge_weight = None
                else:
                    T_ew = T.get_weighted_edges()
                    T_edge_index = torch.tensor(list(map(lambda tp: tp[0:2], T_ew)), dtype=torch.int64).t() if len(T_ew) > 0 else torch.zeros(2, 0, dtype=torch.int64)
                    T_edge_weight = torch.tensor(list(map(lambda tp: tp[2], T_ew)), dtype=torch.float32) if len(T_ew) > 0 else torch.zeros(0, dtype=torch.int64)

                # continue from here...

                subgraph_edge_index, subgraph_edge_weight = T_edge_index, T_edge_weight
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_weight)

                yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                      subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                      subgraph_edge_index.to(device), None if (subgraph_edge_weight is None) else subgraph_edge_weight.to(device), subgraph_rv.to(device)

        return generator()


def rand_split(x, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    shuffled_x = np.random.permutation(x)
    n = len(shuffled_x)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(shuffled_x[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


def time_split(n, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    idx = np.arange(n)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(idx[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


def simplex_coordinates(m):
    x = torch.zeros([m, m+1])
    for j in range(m):
        x[j,j] = 1.0

    a = (1.0 - np.sqrt(float(1+m))) / float(m)
    for i in range(m):
        x[i,m] = a

    c = torch.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(m+1):
            s = s + x[i,j]
        c[i] = s / float(m+1)

    for j in range(m+1):
        for i in range(m):
            x[i,j] = x[i,j] - c[i]

    s = 0.0
    for i in range(m):
        s = s + x[i,0] ** 2
    s = np.sqrt(s)

    for j in range(m+1):
        for i in range(m):
            x[i,j] = x[i,j] / s

    return x.transpose(0,1)


def simplex_coordinates_test(m):
    x = simplex_coordinates(m)

    print("x^{intercal} x")
    print(torch.mm(x, x.transpose(0,1)))


def process_edge_index(num_nodes, edge_index, edge_attr=None):
    def get_undirected(num_nodes, edge_index, edge_attr):
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_attr = None if (edge_attr is None) else torch.cat([edge_attr, edge_attr], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op='max')
        return edge_index, edge_attr

    def sort_edge(num_nodes, edge_index):
        idx = edge_index[0]*num_nodes+edge_index[1]
        sid, perm = idx.sort()
        assert sid.unique_consecutive().shape == sid.shape
        return edge_index[:,perm], perm

    # process edge_attr
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if not is_undirected(edge_index):
        edge_index, edge_attr = get_undirected(num_nodes, edge_index, edge_attr)
    edge_index, od = sort_edge(num_nodes, edge_index)
    _, rv = sort_edge(num_nodes, edge_index.flip(dims=[0]))
    assert torch.all(edge_index[:, rv] == edge_index.flip(dims=[0]))

    return edge_index, (None if edge_attr is None else edge_attr[...,od]), rv


def load_citation(name='Cora', transform=None, split=None):
    data = Planetoid(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    if split is not None:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)

        data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
        data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
        data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_coauthor(name='CS', transform=None, split=[0.3, 0.2, 0.5]):
    data = Coauthor(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_wikipedia(name='Squirrel', transform=None, split=0):
    data = WikipediaNetwork(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    if type(split) == int:
        data.train_mask = data.train_mask[:, split]
        data.val_mask = data.val_mask[:, split]
        data.test_mask = data.test_mask[:, split]
    elif type(split) == list:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)

        data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
        data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
        data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    data.y = data.y.long()

    return data if (transform is None) else transform(data)


def load_county_facebook(transform=None, split=[0.3, 0.2, 0.5], normalize=True):
    dat = pd.read_csv('datasets/county_facebook/dat.csv')
    adj = pd.read_csv('datasets/county_facebook/adj.csv')

    x = torch.tensor(dat.values[:, :9], dtype=torch.float32)
    if normalize:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.tensor(dat.values[:, 9] < dat.values[:, 10], dtype=torch.int64)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_sexual_interaction(transform=None, split=[0.3, 0.2, 0.5]):
    dat = pd.read_csv('datasets/sexual_interaction/dat.csv', header=None)
    adj = pd.read_csv('datasets/sexual_interaction/adj.csv', header=None)

    y = torch.tensor(dat.values[:, 0], dtype=torch.int64)
    x = torch.tensor(dat.values[:, 1:], dtype=torch.float32)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_animals(homo_ratio=0.5, transform=None, split=[0.3, 0.2, 0.5]):
    cat_images = torch.load('datasets/animals/cat_images.pt')
    dog_images = torch.load('datasets/animals/dog_images.pt')
    panda_images = torch.load('datasets/animals/panda_images.pt')

    x = torch.cat((cat_images, dog_images, panda_images), dim=0)
    y = torch.cat((torch.zeros(cat_images.shape[0], dtype=torch.int64),
                   torch.ones(dog_images.shape[0], dtype=torch.int64),
                   torch.ones(panda_images.shape[0], dtype=torch.int64)*2), dim=0)
    edge_index = stochastic_blockmodel_graph([cat_images.shape[0], dog_images.shape[0], panda_images.shape[0]],
                                             torch.tensor([[homo_ratio, 1.0-homo_ratio, 1.0-homo_ratio],
                                                           [1.0-homo_ratio, homo_ratio, 1.0-homo_ratio],
                                                           [1.0-homo_ratio, 1.0-homo_ratio, homo_ratio]])*0.025)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_cats_dogs(homo_ratio=0.5, transform=None, split=[0.3, 0.2, 0.5]):
    cat_images = torch.load('datasets/cats_dogs/cat_images.pt')
    dog_images = torch.load('datasets/cats_dogs/dog_images.pt')

    x = torch.cat((cat_images, dog_images), dim=0)
    y = torch.cat((torch.zeros(cat_images.shape[0], dtype=torch.int64), torch.ones(dog_images.shape[0], dtype=torch.int64)), dim=0)
    edge_index = stochastic_blockmodel_graph([cat_images.shape[0], dog_images.shape[0]],
                                             torch.tensor([[homo_ratio, 1.0-homo_ratio],
                                                           [1.0-homo_ratio, homo_ratio]])*0.003)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def load_ogbn(name='products', transform=None, split=None):
    dataset = PygNodePropPredDataset(root='datasets', name='ogbn-{}'.format(name))
    num_nodes = dataset[0].x.shape[0]

    if split is not None:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    else:
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    data = dataset[0]
    data.y = data.y.reshape(-1)
    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_jpmc_fraud():

    def time_encoding(dt_str):
        dt = datetime.fromisoformat(dt_str)
        soy = datetime(dt.year, 1, 1, 0, 0, 0)
        som = datetime(dt.year, dt.month, 1, 0, 0, 0)
        sow = datetime(dt.year, dt.month, dt.day, 0, 0, 0) - timedelta(days=dt.weekday())
        sod = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        tfy = (dt - soy) / timedelta(days=366)
        tfm = (dt - som) / timedelta(days=31)
        tfw = (dt - sow) / timedelta(days=7)
        tfd = (dt - sod) / timedelta(days=1)
        tfs = np.array([tfy, tfm, tfw, tfd])
        emb = np.concatenate((np.sin(2 * np.pi * tfs), np.cos(2 * np.pi * tfs)))
        return emb

    dat = pd.read_csv('datasets/aml/jpmc_synthetic.csv', delimiter=',', skipinitialspace=True)

    # get time embedding
    time_emb = torch.tensor(dat['Time_step'].apply(time_encoding).array)

    # get logarithmic transformed USD amount
    log_usd = torch.tensor(np.log(dat['USD_Amount']).array)

    # get one-hot encoding for sender country and bene country
    all_countries = pd.concat((dat['Sender_Country'], dat['Bene_Country'])).unique()
    cty2cid = {country: cid for (cid, country) in enumerate(all_countries)}
    sender_cid = torch.tensor(list(dat['Sender_Country'].apply(lambda x: cty2cid[x])))
    bene_cid = torch.tensor(list(dat['Bene_Country'].apply(lambda x: cty2cid[x])))
    sender_cty_emb = F.one_hot(sender_cid, len(all_countries))
    bene_cty_emb = F.one_hot(bene_cid, len(all_countries))

    # get one-hot encoding for sector
    all_sid = dat['Sector'].unique()
    all_sid.sort()
    assert np.all(all_sid == np.arange(len(all_sid)))
    sid = torch.tensor(dat['Sector'])
    sct_emb = F.one_hot(sid, len(all_sid))

    # get one-hot encoding for transaction type
    all_tst = dat['Transaction_Type'].unique()
    tst2tid = {transaction: tid for (tid, transaction) in enumerate(all_tst)}
    tid = torch.tensor(list(dat['Transaction_Type'].apply(lambda x: tst2tid[x])))
    tst_emb = F.one_hot(tid, len(all_tst))

    feature = torch.cat((time_emb, log_usd.view(-1,1), sender_cty_emb, bene_cty_emb, sct_emb, tst_emb), dim=-1)
    label = torch.tensor(dat['Label'].array)

    return feature.numpy(), label.numpy(), dat[['Sender_Id', 'Bene_Id']]


def pad_sequence(sequences, batch_first=False, padding_value=0.0):

    trailing_dims = sequences[0].shape[1:]
    max_len = max([s.size(0) for s in sequences])
    mask_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    out_dims = mask_dims + trailing_dims

    mask_tensor = torch.zeros(mask_dims, dtype=torch.bool)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            mask_tensor[i, :length] = True
            out_tensor[i, :length, ...] = tensor
        else:
            mask_tensor[:length, i] = True
            out_tensor[:length, i, ...] = tensor

    return out_tensor, mask_tensor


def preprocess_rnn_jpmc_fraud(feature, label, info, previous_label_as_feature=False):
    info['index'] = info.index
    group = info[['index', 'Sender_Id']].groupby('Sender_Id')['index'].apply(np.array).reset_index(name='indices')

    group_feature = [torch.tensor(feature[idx], dtype=torch.float32) for idx in group['indices']]
    group_label = [torch.tensor(label[idx], dtype=torch.int64) for idx in group['indices']]
    
    feature_padded, feature_mask = pad_sequence(group_feature, batch_first=True)
    label_padded, label_mask = pad_sequence(group_label, batch_first=True)
    assert torch.all(feature_mask == label_mask)

    if previous_label_as_feature:
        previous_label = torch.cat((label_padded.new_full((label_padded.shape[0], 1), fill_value=1), label_padded[:, :-1]), dim=1).unsqueeze(dim=-1)
        feature_padded = torch.cat((feature_padded, previous_label), dim=2)

    return feature_padded, label_padded, feature_mask


def preprocess_gnn_jpmc_fraud(feature, label, info, transform=None, split=[0.3, 0.2, 0.5]):
    last_seen = defaultdict(lambda: -1)
    edge_idx_list = []
    for i in range(info.shape[0]):
        sender_id, bene_id = info['Sender_Id'][i], info['Bene_Id'][i]
        assert sender_id != bene_id
        if not pd.isna(sender_id):
            if last_seen[sender_id] != -1:
                edge_idx_list.append([last_seen[sender_id], i])
            last_seen[sender_id] = i
        if not pd.isna(bene_id):
            if last_seen[bene_id] != -1:
                edge_idx_list.append([last_seen[bene_id], i])
            last_seen[bene_id] = i

    edge_index = to_undirected(remove_self_loops(torch.tensor(edge_idx_list).transpose(0,1))[0])
    data = Data(x=torch.tensor(feature, dtype=torch.float32), y=torch.tensor(label, dtype=torch.int64), edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter(0, torch.tensor(test_idx), True)

    data.edge_index, data.edge_weight, data.rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def acc(score, y, mask):
    return int(score.max(dim=-1)[1][mask].eq(y[mask]).sum().item()) / int(mask.sum())


def log_normalize(log_x):
    return log_x - torch.logsumexp(log_x, -1, keepdim=True)


def subsample(mask, p=0.10):
    return torch.logical_and(mask, torch.rand(mask.shape) < p)


def adj_normalize(adj_t, add_self_loops=True, normalization='sym'):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.0)
    if add_self_loops:
        adj_t = torch_sparse.fill_diag(adj_t, 1.0)

    deg = torch_sparse.sum(adj_t, dim=1)
    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        norm_adj_t = torch_sparse.mul(torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1)), deg_inv_sqrt.view(1, -1))
    elif normalization == 'rw':
        deg_inv = deg.pow_(-1.0)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        norm_adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
    else:
        raise Exception('unexpected normalization type')

    return norm_adj_t


def nll_loss(projection, target, reduction='mean'):
    loss = F.nll_loss(torch.log_softmax(projection, dim=-1), target, reduction=reduction)
    return loss


def exp_loss(projection, target, reduction='mean'):
    loss = (projection - projection.gather(1, target.view(-1, 1))).exp()
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise Exception('unexpected reduction type')
    return loss


def create_outpath(dataset, model_name):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, model_name + '-' + dataset + '-' + '{:05d}'.format(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath
