import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch_sparse
from torch_geometric.utils import remove_self_loops, to_undirected, is_undirected
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
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


def process_edge_index(edge_index):
    assert is_undirected(edge_index)
    edge_index = edge_index.unique(dim=1, sorted=True)
    _, rv = edge_index.flip(dims=[0]).unique(dim=1, sorted=True, return_inverse=True)
    assert torch.all(edge_index[:, rv] == edge_index.flip(dims=[0]))

    return edge_index, rv


def load_citation(name='Cora', transform=None, split='original'):
    data = Planetoid(root='/tmp/{}'.format(name), name=name)[0]

    if split != 'original':
        train_idx, val_idx, test_idx = rand_split(data.x.shape[0], split)

        data.train_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(train_idx), True)
        data.val_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(val_idx), True)
        data.test_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.rv = process_edge_index(data.edge_index)

    return data if (transform is None) else transform(data)


def load_county_facebook(transform=None, split=[0.3, 0.2, 0.5], normalize=True):
    dat = pd.read_csv('dataset/county_facebook/dat.csv')
    adj = pd.read_csv('dataset/county_facebook/adj.csv')

    x = torch.tensor(dat.values[:, :9], dtype=torch.float32)
    if normalize:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.tensor(dat.values[:, 9] < dat.values[:, 10], dtype=torch.int64)
    edge_index = to_undirected(remove_self_loops(torch.transpose(torch.tensor(adj.values), 0, 1))[0])

    data = Data(x=x, y=y, edge_index=edge_index)
    train_idx, val_idx, test_idx = rand_split(x.shape[0], split)

    data.train_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(train_idx), True)
    data.val_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(val_idx), True)
    data.test_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, torch.tensor(test_idx), True)

    data.edge_index, data.rv = process_edge_index(data.edge_index)

    return data if (transform is None) else transform(data)


def load_ogbn_product(transform=None):
    dataset = PygNodePropPredDataset(name='ogbn-products')

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    data = dataset[0]
    data.y = data.y.reshape(-1)
    data.train_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(data.x.shape[0], dtype=bool).scatter_(0, test_idx, True)
    data.edge_index = remove_self_loops(data.edge_index)[0]

    data.edge_index, data.rv = process_edge_index(data.edge_index)

    return data if (transform is None) else transform(data)


def acc(score, y, mask):
    return int(score.max(dim=-1)[1][mask].eq(y[mask]).sum().item()) / int(mask.sum())


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
