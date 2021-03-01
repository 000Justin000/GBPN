import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, subgraph, remove_self_loops, to_undirected, is_undirected, contains_self_loops, stochastic_blockmodel_graph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset


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

    edge_index = to_undirected(remove_self_loops(edge_index)[0], num_nodes)
    edge_index, od = sort_edge(num_nodes, edge_index)
    _, rv = sort_edge(num_nodes, edge_index.flip(dims=[0]))
    # assert not contains_self_loops(edge_index)
    # assert is_undirected(edge_index)
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
                print(subgraph_nodes.shape[0])
                for _ in range(num_hops):
                    _, subgraph_nodes = self.adj_t.sample_adj(subgraph_nodes, size, replace=False)
                    print(subgraph_nodes.shape[0])
                subgraph_size = subgraph_nodes.shape[0]

                assert torch.all(subgraph_nodes[:batch_size] == batch_nodes)
                subgraph_edge_index, subgraph_edge_weight = subgraph(subgraph_nodes, self.edge_index, self.edge_weight, relabel_nodes=True)
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv = process_edge_index(subgraph_nodes.shape[0], subgraph_edge_index, subgraph_edge_weight)
                print(subgraph_edge_index.shape)

                yield batch_size, batch_nodes.to(device), self.x[batch_nodes].to(device), self.y[batch_nodes].to(device), self.deg[batch_nodes].to(device), \
                      subgraph_size, subgraph_nodes.to(device), self.x[subgraph_nodes].to(device), self.y[subgraph_nodes].to(device), self.deg[subgraph_nodes].to(device), \
                      subgraph_edge_index.to(device), None if (subgraph_edge_weight is None) else subgraph_edge_weight.to(device), subgraph_rv.to(device)

        return generator()


def get_scaling(deg0, deg1):
    assert deg0.shape == deg1.shape
    scaling = torch.ones(deg0.shape[0]).to(deg0.device)
    scaling[deg1 != 0] = (deg0 / deg1)[deg1 != 0]
    return scaling
