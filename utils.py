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
from torch_geometric.utils import degree, subgraph, remove_self_loops, to_undirected, contains_self_loops, is_undirected, stochastic_blockmodel_graph, k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
from datetime import datetime, timedelta
from collections import defaultdict


class MultiOptimizer:

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, activation=nn.ReLU(), dropout_p=0.0):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_layers >= 2:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers-2)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of layers must be positive')
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = self.dropout(x)
            x = linear(x)
            if i < self.num_layers-1:
                x = self.activation(x)
        return x


class GMLP(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, activation=nn.ReLU(), dropout_p=0.0):
        super(GMLP, self).__init__()
        self.mlp = MLP(dim_in, dim_out, dim_hidden, num_layers, activation, dropout_p)

    def forward(self, x, edge_index, **kwargs):
        return F.log_softmax(self.mlp(x), dim=-1)

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, **kwargs):
        x_all = torch.zeros(sampler.num_nodes, self.mlp.linears[-1].out_features, dtype=torch.float32)
        for batch_size, batch_nodes, _, _, _, \
            _, _, subgraph_x, _, _, \
            subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=0):
            x_all[batch_nodes] = self.forward(subgraph_x.to(device), subgraph_edge_index.to(device))[:batch_size].cpu()
        return x_all


class SGC(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, dropout_p=0.0):
        super(SGC, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_in, dim_hidden))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(dim_hidden, dim_hidden))
        self.convs.append(GCNConv(dim_hidden, dim_out))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = self.dropout(x)
            x = conv(x, edge_index)
            if i == self.num_layers-1:
                x = F.log_softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, **kwargs):
        assert type(sampler) == FullgraphSampler
        assert max_batch_size == -1
        x_all = torch.zeros(sampler.num_nodes, self.convs[-1].out_channels, dtype=torch.float32)
        for batch_size, batch_nodes, _, _, _, \
            _, _, subgraph_x, _, _, \
            subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=self.num_layers):
            x_all[batch_nodes] = self.forward(subgraph_x.to(device), subgraph_edge_index.to(device))[:batch_size].cpu()
        return x_all


class GCN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, activation=nn.ReLU(), dropout_p=0.0):
        super(GCN, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_in, dim_hidden))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(dim_hidden, dim_hidden))
        self.convs.append(GCNConv(dim_hidden, dim_out))
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = self.dropout(x)
            x = conv(x, edge_index)
            if i < self.num_layers-1:
                x = self.activation(x)
            else:
                x = F.log_softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, **kwargs):
        assert type(sampler) == FullgraphSampler
        assert max_batch_size == -1
        x_all = torch.zeros(sampler.num_nodes, self.convs[-1].out_channels, dtype=torch.float32)
        for batch_size, batch_nodes, _, _, _, \
            _, _, subgraph_x, _, _, \
            subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=self.num_layers):
            x_all[batch_nodes] = self.forward(subgraph_x.to(device), subgraph_edge_index.to(device))[:batch_size].cpu()
        return x_all


class SAGE(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, activation=nn.ReLU(), dropout_p=0.0):
        super(SAGE, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(dim_in, dim_hidden))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(dim_hidden, dim_hidden))
        self.convs.append(SAGEConv(dim_hidden, dim_out))
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = self.dropout(x)
            x = conv(x, edge_index)
            if i < self.num_layers-1:
                x = self.activation(x)
            else:
                x = F.log_softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, **kwargs):
        x_all_ = sampler.x
        for i, conv in enumerate(self.convs):
            x_all = torch.zeros(sampler.num_nodes, conv.out_channels, dtype=torch.float32)
            for _, batch_nodes, _, _, _, \
                _, subgraph_nodes, _, _, _, \
                subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=1):
                x, subgraph_edge_index = x_all_[subgraph_nodes].to(device), subgraph_edge_index.to(device)
                x = self.dropout(x)
                x = conv(x, subgraph_edge_index)
                if i < self.num_layers-1:
                    x = self.activation(x)
                else:
                    x = F.log_softmax(x, dim=-1)
                x_all[batch_nodes] = x[:batch_nodes.shape[0]].cpu()
            x_all_ = x_all
        return x_all


class GAT(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=128, num_layers=2, num_heads=1, activation=nn.ELU(), dropout_p=0.0):
        super(GAT, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.convs.append(GATConv(dim_in, dim_hidden, heads=num_heads))
        self.skips.append(nn.Linear(dim_in, dim_hidden*num_heads))
        for _ in range(num_layers-2):
            self.convs.append(GATConv(dim_hidden*num_heads, dim_hidden, heads=num_heads))
            self.skips.append(nn.Linear(dim_hidden*num_heads, dim_hidden*num_heads))
        self.convs.append(GATConv(dim_hidden*num_heads, dim_out, heads=num_heads, concat=False))
        self.skips.append(nn.Linear(dim_hidden*num_heads, dim_out))
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = self.dropout(x)
            x = conv(x, edge_index) + skip(x)
            if i < self.num_layers-1:
                x = self.activation(x)
            else:
                x = F.log_softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, **kwargs):
        x_all_ = sampler.x
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x_all = torch.zeros(sampler.num_nodes, skip.out_features, dtype=torch.float32)
            for _, batch_nodes, _, _, _, \
                _, subgraph_nodes, _, _, _, \
                subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=1):
                x, subgraph_edge_index = x_all_[subgraph_nodes].to(device), subgraph_edge_index.to(device)
                x = self.dropout(x)
                x = conv(x, subgraph_edge_index) + skip(x)
                if i < self.num_layers-1:
                    x = self.activation(x)
                else:
                    x = F.log_softmax(x, dim=-1)
                x_all[batch_nodes] = x[:batch_nodes.shape[0]].cpu()
            x_all_ = x_all
        return x_all


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
        self.param = nn.Parameter(torch.eye(n_channels) * 0.0)
        # self.PM_Net = MLP(n_channels*n_channels, n_channels*n_channels, num_layers=3)
        self.n_channels = n_channels

    def get_logH(self):
        PM = self.param
        # PM = self.PM_Net(self.param.reshape(-1)).reshape((self.n_channels, self.n_channels))
        logH = F.logsigmoid(PM + PM.t())
        return (logH if self.learn_H else F.logsigmoid(torch.zeros(self.n_channels, self.n_channels)).fill_diagonal_(0.0).to(logH.device))

    def forward(self, x, edge_index, edge_weight, info):
        # x has shape [N, n_channels]
        # edge_index has shape [2, E]
        # info has 4 fields: 'log_b0', 'log_msg_', 'edge_rv', 'agg_scaling'
        return self.propagate(edge_index, edge_weight=edge_weight, x=x, info=info)

    def message(self, x_j, edge_weight, info):
        # x_j has shape [E, n_channels]
        if info['log_msg_'] is not None:
            x_j = x_j - info['log_msg_'][info['edge_rv']]
        logC = self.get_logH().unsqueeze(0) * edge_weight.unsqueeze(-1).unsqueeze(-1)
        log_msg = log_normalize(torch.logsumexp(x_j.unsqueeze(-1) + logC, dim=-2))
        info['log_msg_'] = log_msg
        return log_msg

    def update(self, agg_log_msg, info):
        log_b_raw = info['log_b0'] + agg_log_msg
        if info['agg_scaling'] is not None:
            log_b_raw = log_b_raw + (info['agg_scaling']-1.0).unsqueeze(-1) * agg_log_msg.detach()
        log_b = log_normalize(log_b_raw)
        return log_b


class LogEvidentialProb(nn.Module):

    def __init__(self, dim=-1):
        super(LogEvidentialProb, self).__init__()
        self.dim = dim

    def forward(self, x):
        evidence = nn.functional.softplus(x, beta=5.0) + 0.1
        return torch.log(evidence / evidence.sum(dim=self.dim, keepdim=True))


class GBPN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=32, num_layers=0, activation=nn.ReLU(), dropout_p=0.0, learn_H=False):
        super(GBPN, self).__init__()
        self.transform = nn.Sequential(MLP(dim_in, dim_out, dim_hidden=dim_hidden, num_layers=num_layers, activation=activation, dropout_p=dropout_p), nn.LogSoftmax(dim=-1))
        self.bp_conv = BPConv(dim_out, learn_H)

    def forward(self, x, edge_index, edge_weight, edge_rv, phi=None, agg_scaling=None, K=5):
        log_b0 = self.transform(x)
        if phi is not None:
            log_b0 = log_normalize(log_b0 + phi)
        info = {'log_b0': log_b0, 'log_msg_': None, 'edge_rv': edge_rv, 'agg_scaling': agg_scaling}
        log_b = log_b0
        for _ in range(K):
            log_b = self.bp_conv(log_b, edge_index, edge_weight, info)
        return log_b

    @torch.no_grad()
    def inference(self, sampler, max_batch_size, device, phi=None, K=5):
        log_b0_list = []
        for batch_size, _, _, _, _, \
            _, _, subgraph_x, _, _, \
            subgraph_edge_index, _, _, _ in sampler.get_generator(max_batch_size=max_batch_size, num_hops=0):
            log_b0_list.append(self.transform(subgraph_x.to(device))[:batch_size].cpu())
        log_b0 = torch.cat(log_b0_list, dim=0)
        if phi is not None:
            log_b0 = log_normalize(log_b0 + phi)
        log_b_ = log_b0
        log_msg_ = torch.zeros(sampler.edge_index.shape[1], log_b0.shape[1], dtype=torch.float32)
        for _ in range(K):
            log_b = torch.zeros_like(log_b_)
            log_msg = torch.zeros_like(log_msg_)
            for batch_size, batch_nodes, _, _, _, \
                subgraph_size, subgraph_nodes, _, _, _, \
                subgraph_edge_index, subgraph_edge_weight, subgraph_edge_rv, subgraph_edge_oid in sampler.get_generator(max_batch_size=max_batch_size, num_hops=1):
                info = {'log_b0': log_b0[subgraph_nodes].to(device), 'log_msg_': log_msg_[subgraph_edge_oid].to(device), 'edge_rv': subgraph_edge_rv.to(device), 'agg_scaling': None}
                log_b[batch_nodes] = self.bp_conv(log_b_[subgraph_nodes].to(device), subgraph_edge_index.to(device), subgraph_edge_weight.to(device), info)[:batch_size].cpu()
                subgraph_edge_mask = subgraph_edge_index[1] < batch_size
                log_msg[subgraph_edge_oid[subgraph_edge_mask]] = info['log_msg_'][subgraph_edge_mask].cpu()
            log_b_ = log_b
            log_msg_ = log_msg
        return log_b_


class FullgraphSampler:

    def __init__(self, num_nodes, x, y, edge_index, edge_weight, edge_rv):
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_rv = edge_rv
        self.deg = degree(edge_index[1], num_nodes)

    def get_generator(self, mask=None, device='cpu', **kwargs):

        def generator():
            batch_nodes = torch.arange(self.num_nodes, dtype=torch.int64) if (mask is None) else mask.nonzero(as_tuple=True)[0]

            subgraph_nodes = torch.cat((batch_nodes, torch.tensor(list(set(range(self.num_nodes)) - set(batch_nodes.tolist())), dtype=torch.int64)), dim=0)
            subgraph_edge_index, subgraph_edge_oid = subgraph(subgraph_nodes, self.edge_index, torch.arange(self.edge_index.shape[1]), relabel_nodes=True, num_nodes=self.num_nodes)
            subgraph_edge_index, subgraph_edge_oid, subgraph_edge_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_oid)

            subgraph_edge_weight = self.edge_weight[subgraph_edge_oid]

            batch_size = batch_nodes.shape[0]
            subgraph_size = subgraph_nodes.shape[0]

            yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                  subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                  subgraph_edge_index.to(device), subgraph_edge_weight.to(device), subgraph_edge_rv.to(device), subgraph_edge_oid.to(device)

        return generator()


class SubtreeSampler:

    def __init__(self, num_nodes, x, y, edge_index, edge_weight, edge_rv):
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_rv = edge_rv
        self.G = nx.Graph(num_nodes)
        self.G.add_edges_from(torch.cat((edge_index, torch.arange(edge_index.shape[1]).reshape(1,-1)), dim=0).transpose(0,1).numpy())
        self.deg = degree(edge_index[1], num_nodes)

    def get_generator(self, mask=None, shuffle=True, max_batch_size=-1, num_hops=0, num_samples=-1, device='cpu'):
        idx = torch.arange(self.num_nodes, dtype=torch.int64) if (mask is None) else mask.nonzero(as_tuple=True)[0]
        if shuffle:
            idx = idx[torch.randperm(idx.shape[0])]
        if max_batch_size == -1:
            max_batch_size = self.num_nodes

        n_batch = math.ceil(idx.shape[0] / max_batch_size)

        def generator():
            for batch_nodes in idx.chunk(n_batch):
                batch_size = batch_nodes.shape[0]

                T = nx.sample_subtree(self.G, batch_nodes.tolist(), num_hops, num_samples)
                subgraph_nodes = torch.tensor(T.get_nodes(), dtype=torch.int64)
                subgraph_size = subgraph_nodes.shape[0]
                T_edges = torch.tensor(T.get_edges(), dtype=torch.int64)
                subgraph_edge_index = T_edges[:,0:2].t() if T_edges.shape[0] > 0 else torch.zeros(2, 0, dtype=torch.int64)
                subgraph_edge_oid = T_edges[:,2] if T_edges.shape[0] > 0 else torch.zeros(0, dtype=torch.int64)

                assert subgraph_size == subgraph_edge_index.shape[1]//2 + batch_size
                assert torch.all(subgraph_nodes[subgraph_edge_index.reshape(-1)] == self.edge_index[:,subgraph_edge_oid].reshape(-1))

                subgraph_edge_index, subgraph_edge_oid, subgraph_edge_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_oid)
                subgraph_edge_weight = self.edge_weight[subgraph_edge_oid]

                yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                      subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                      subgraph_edge_index.to(device), subgraph_edge_weight.to(device), subgraph_edge_rv.to(device), subgraph_edge_oid.to(device)

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
    if (edge_index.shape[1] > 0) and (not is_undirected(edge_index)):
        edge_index, edge_attr = get_undirected(num_nodes, edge_index, edge_attr)
    edge_index, od = sort_edge(num_nodes, edge_index)
    _, edge_rv = sort_edge(num_nodes, edge_index.flip(dims=[0]))
    assert torch.all(edge_index[:, edge_rv] == edge_index.flip(dims=[0]))

    return edge_index, (None if edge_attr is None else edge_attr[...,od]), edge_rv


def load_citation(name='Cora', transform=None, split=None):
    data = Planetoid(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    if split is not None:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)
        train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

        data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
        data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
        data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_coauthor(name='CS', transform=None, split=[0.3, 0.2, 0.5]):
    data = Coauthor(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

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
        train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

        data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
        data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
        data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    data.y = data.y.long()

    return data if (transform is None) else transform(data)


def load_county_facebook(transform=None, split=[0.3, 0.2, 0.5], normalize=True):
    dat = pd.read_csv('datasets/county_facebook/dat.csv')
    adj = pd.read_csv('datasets/county_facebook/adj.csv')

    x = torch.tensor(dat.values[:, 0:9], dtype=torch.float32)
    if normalize:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.tensor(dat.values[:, 9] < dat.values[:, 10], dtype=torch.int64)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_sexual_interaction(transform=None, split=[0.3, 0.2, 0.5]):
    dat = pd.read_csv('datasets/sexual_interaction/dat.csv', header=None)
    adj = pd.read_csv('datasets/sexual_interaction/adj.csv', header=None)

    y = torch.tensor(dat.values[:, 0], dtype=torch.int64)
    x = torch.tensor(dat.values[:, 1:21], dtype=torch.float32)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_animal2(homo_ratio=0.5, transform=None, split=[0.3, 0.2, 0.5]):
    cat_images = torch.load('datasets/animal2/cat_images.pt')
    dog_images = torch.load('datasets/animal2/dog_images.pt')

    x = torch.cat((cat_images, dog_images), dim=0)
    y = torch.cat((torch.zeros(cat_images.shape[0], dtype=torch.int64), torch.ones(dog_images.shape[0], dtype=torch.int64)), dim=0)
    edge_index = stochastic_blockmodel_graph([cat_images.shape[0], dog_images.shape[0]],
                                             torch.tensor([[homo_ratio, 1.0-homo_ratio],
                                                           [1.0-homo_ratio, homo_ratio]])*0.003)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def load_animal3(homo_ratio=0.5, transform=None, split=[0.3, 0.2, 0.5]):
    cat_images = torch.load('datasets/animal3/cat_images.pt')
    dog_images = torch.load('datasets/animal3/dog_images.pt')
    panda_images = torch.load('datasets/animal3/panda_images.pt')

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
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_ogbn(name='products', transform=None, split=None):
    dataset = PygNodePropPredDataset(root='datasets', name='ogbn-{}'.format(name))
    num_nodes = dataset[0].x.shape[0]

    if split is not None:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)
        train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)
    else:
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    data = dataset[0]
    data.y = data.y.reshape(-1)
    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_jpmc_payment(dataset_id):

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

    dat = pd.read_csv('datasets/aml/jpmc/payment{}.csv'.format(dataset_id), delimiter=',', skipinitialspace=True)

    # get time embedding
    time_emb = torch.tensor(dat['Time_step'].apply(time_encoding).array)

    # get logarithmic transformed USD amount
    log_usd = torch.tensor(np.log(dat['USD_Amount'] + 1.0).array)

    # get one-hot encoding for sender country and bene country
    all_countries = pd.concat((dat['Sender_Country'], dat['Bene_Country'])).unique()
    cty2cid = {country: cid for (cid, country) in enumerate(all_countries)}
    sender_cid = torch.tensor(list(dat['Sender_Country'].apply(lambda x: cty2cid[x])))
    bene_cid = torch.tensor(list(dat['Bene_Country'].apply(lambda x: cty2cid[x])))
    sender_cty_emb = F.one_hot(sender_cid, len(all_countries))
    bene_cty_emb = F.one_hot(bene_cid, len(all_countries))

    # get one-hot encoding for sector
    if 'Sector' in dat.columns:
        all_sid = dat['Sector'].unique()
        all_sid.sort()
        assert np.all(all_sid == np.arange(len(all_sid)))
        sid = torch.tensor(dat['Sector'])
        sct_emb = F.one_hot(sid, len(all_sid))
    else:
        sct_emb = torch.zeros(dat.shape[0], 0)

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


def preprocess_rnn_jpmc_payment(feature, label, info, previous_label_as_feature=False):
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


def preprocess_gnn_jpmc_payment(feature, label, info, transform=None, split=[0.3, 0.2, 0.5]):
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

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def load_elliptic_bitcoin(transform=None, split=None):
    dat = pd.read_csv('datasets/aml/elliptic/elliptic_txs_features.csv', header=None).sort_values(0)
    cls = pd.read_csv('datasets/aml/elliptic/elliptic_txs_classes.csv', header=0).sort_values('txId')
    edge_ids = pd.read_csv('datasets/aml/elliptic/elliptic_txs_edgelist.csv', header=0)

    assert (dat[0] == cls['txId']).all()
    id2num = dict(map(reversed, enumerate(dat[0])))

    features = torch.tensor(dat.iloc[:, 2:].to_numpy(), dtype=torch.float32)
    labels = torch.tensor(list(map(lambda x: -1 if (x == 'unknown') else (0 if x == '2' else 1), cls['class'])))

    edge_num1 = list(map(lambda x: id2num[x], edge_ids['txId1']))
    edge_num2 = list(map(lambda x: id2num[x], edge_ids['txId2']))
    edge_index = torch.stack((torch.tensor(edge_num1), torch.tensor(edge_num2)), dim=0)

    num_nodes = features.shape[0]
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data = Data(x=features, y=labels, edge_index=edge_index)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True) & (labels != -1)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True) & (labels != -1)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True) & (labels != -1)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def load_ising(transform=None, split=[0.3, 0.2, 0.5], interaction='+', dataset_id=0):
    edge_index = torch.tensor((pd.read_csv('datasets/ising{}/adj'.format(interaction), sep='\t', header=None)-1).to_numpy().T)
    features = torch.tensor(pd.read_csv('datasets/ising{}/coord'.format(interaction), sep='\t', header=None).to_numpy(), dtype=torch.float32)
    labels = torch.tensor(pd.read_csv('datasets/ising{}/label'.format(interaction), sep='\t', header=None).to_numpy(), dtype=torch.int64)[:,dataset_id]
    labels[labels == -1] = 0

    num_nodes = features.shape[0]
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data = Data(x=features, y=labels, edge_index=edge_index)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, None)

    return data if (transform is None) else transform(data)


def load_mrf(transform=None, split=[0.3, 0.2, 0.5], interaction='+', dataset_id=0):
    edge_index = torch.tensor((pd.read_csv('datasets/mrf{}/adj'.format(interaction), sep='\t', header=None)-1).to_numpy().T)
    features = torch.tensor(pd.read_csv('datasets/mrf{}/coord'.format(interaction), sep='\t', header=None).to_numpy(), dtype=torch.float32)
    labels = torch.tensor(pd.read_csv('datasets/mrf{}/label'.format(interaction), sep='\t', header=None).to_numpy(), dtype=torch.int64)[:,dataset_id]

    num_nodes = features.shape[0]
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data = Data(x=features, y=labels, edge_index=edge_index)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, None)

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
