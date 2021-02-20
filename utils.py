import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch_sparse
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops, to_undirected, is_undirected, contains_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.CELU(), dropout_p=0.0):
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


class SGConv(MessagePassing):
    def __init__(self, dim_in, dim_out, K=1):
        super(SGConv, self).__init__(aggr='add')
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.K = K
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, edge_index):
        edge_index, edge_weight = gcn_norm(edge_index, num_nodes=x.shape[0], add_self_loops=True, dtype=x.dtype)
        x = self.linear(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class SGC(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SGC, self).__init__()
        self.conv = SGConv(dim_in, dim_out, K=2)

    def forward(self, x, edge_index, **kwargs):
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, dropout_p=0.0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden, cached=True)
        self.conv2 = GCNConv(dim_hidden, dim_out, cached=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GAT(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128, dropout_p=0.0, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_in, dim_hidden, heads=num_heads, dropout=dropout_p)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(dim_hidden*num_heads, dim_out, heads=num_heads, concat=False, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, **kwargs):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


def rand_split(x, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    shuffled_x = np.random.permutation(x)
    n = len(shuffled_x)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(shuffled_x[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


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
    def sort_edge(num_nodes, edge_index):
        idx = edge_index[0]*num_nodes+edge_index[1]
        sid, perm = idx.sort()
        assert sid.unique_consecutive().shape == sid.shape
        return edge_index[:,perm], perm

    edge_index = to_undirected(remove_self_loops(edge_index)[0])
    edge_index, od = sort_edge(num_nodes, edge_index)
    _, rv = sort_edge(num_nodes, edge_index.flip(dims=[0]))
    assert not contains_self_loops(edge_index)
    assert is_undirected(edge_index)
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

    data.y = data.y.type(torch.LongTensor)

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


def create_outpath(dataset):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, 'dataset:'+dataset + '-' + 'pid:'+str(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath
