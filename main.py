import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import sort_edge_index, is_undirected
from utils import *
import matplotlib.pyplot as plt

compute_device = 'cuda'
storage_device = 'cpu'


def log_normalize(log_x):
    return log_x - torch.logsumexp(log_x, -1, keepdim=True)


class BPConv(MessagePassing):
    def __init__(self, n_channels):
        super(BPConv, self).__init__(aggr='add')
        self.T = nn.Parameter(torch.randn(n_channels, n_channels) / (n_channels**0.5))
        self.n_channels = n_channels

    def log_H(self, learned=True):
        if learned:
            T = self.T
            s = (T * T).sum(dim=1)
            S = s.view(-1, 1) + s.view(1, -1) - 2*torch.mm(T, T.transpose(0, 1))
        else:
            S = (torch.ones((self.n_channels, self.n_channels))-torch.eye(self.n_channels))
        return -S

    def forward(self, x, edge_index, info):
        # x has shape [N, n_channels]
        # edge_index has shape [2, E]
        # info has three fields: 'log_b0', 'log_msg_', 'rv'
        return self.propagate(edge_index, x=x, info=info)

    def message(self, x_j, info):
        # x_j has shape [E, n_channels]
        log_msg_raw = torch.logsumexp((x_j - info['log_msg_'][info['rv']]).unsqueeze(-1) + self.log_H().unsqueeze(0), dim=-2)
        log_msg = log_normalize(log_msg_raw)
        info['log_msg_'] = log_msg
        return log_msg

    def update(self, inputs, info):
        log_b_raw = inputs + info['log_b0']
        log_b = log_normalize(log_b_raw)
        return log_b


class BPGNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=0, dropout_p=0.0):
        super(BPGNN, self).__init__()
        self.transform = nn.Sequential(MLP(dim_in, dim_out, dim_hidden, num_hidden, nn.ReLU(), dropout_p), nn.LogSoftmax(dim=-1))
        self.conv = BPConv(dim_out)

    def forward(self, x, edge_index, rv, fixed=None):
        log_b0 = self.transform(x)
        num_classes = log_b0.shape[-1]
        info = {'log_b0': log_b0, 'log_msg_': torch.ones(edge_index.shape[1], num_classes) * (-np.log(num_classes)), 'rv': rv}
        log_b = log_b0
        for _ in range(5):
            log_b = self.conv(log_b, edge_index, info)
            if fixed is not None:
                log_b[fixed[0]] = log_normalize(F.one_hot(fixed[1])*5.0)

        return log_b


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden, cached=True)
        self.conv2 = GCNConv(dim_hidden, dim_out, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


transform = None
# data = load_citation('Cora', transform=transform)
# data = load_citation('CiteSeer', transform=transform)
# data = load_citation('PubMed', transform=transform)
# data = load_citation('Cora', transform=transform, split=[0.6,0.2,0.2])
# data = load_citation('CiteSeer', transform=transform, split=[0.6,0.2,0.2])
data = load_citation('PubMed', transform=transform, split=[0.6,0.2,0.2])
# data = load_county_facebook(transform=transform, split=[0.4,0.1,0.5])

edge_index, rv = data.edge_index, data.rv
x, y = data.x, data.y
num_nodes, num_features = x.shape
num_classes = len(torch.unique(y))
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

gnn = BPGNN(num_features, num_classes, 128, 0, 0.2)

optimizer = torch.optim.AdamW([{'params': gnn.parameters()}], lr=1.0e-2, weight_decay=2.5e-4)
for epoch in range(300):
    optimizer.zero_grad()
    b = gnn(x, edge_index, rv)
    loss = F.nll_loss(b[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    print('step: {:3d}, train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(epoch, acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)

gnn = gnn.eval()
with torch.no_grad():
    b = gnn(x, edge_index, rv)
    print('train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)
    b = gnn(x, edge_index, rv, (train_mask, y[train_mask]))
    print('train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)

print('finished')