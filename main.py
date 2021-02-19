import sys
import subprocess
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import sort_edge_index, is_undirected, subgraph
from utils import *
import matplotlib
import matplotlib.pyplot as plt


class SumConv(MessagePassing):
    def __init__(self):
        super(SumConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


class BPConv(MessagePassing):
    def __init__(self, n_channels, learn_coupling=False):
        super(BPConv, self).__init__(aggr='add')
        self.learn_coupling = learn_coupling
        dim_param = n_channels*(n_channels+1)//2
        self.param = nn.Parameter(torch.zeros(dim_param))
        self.n_channels = n_channels

    def get_logH(self):
        logT = torch.zeros(self.n_channels, self.n_channels).to(self.param.device)
        rid, cid = torch.tril_indices(self.n_channels, self.n_channels, 0)
        logT[rid, cid] = F.logsigmoid(self.param * 10.0)
        logH = (logT + logT.transpose(0,1).triu(1))
        return (logH if self.learn_coupling else logH.detach())

    def forward(self, x, edge_index, info):
        # x has shape [N, n_channels]
        # edge_index has shape [2, E]
        # info has 3 fields: 'log_b0', 'log_msg_', 'rv'
        return self.propagate(edge_index, x=x, info=info)

    def message(self, x_j, info):
        # x_j has shape [E, n_channels]
        log_msg_raw = torch.logsumexp((x_j - info['log_msg_'][info['rv']]).unsqueeze(-1) + self.get_logH().unsqueeze(0), dim=-2)
        log_msg = log_normalize(log_msg_raw)
        info['log_msg_'] = log_msg
        return log_msg

    def update(self, inputs, info):
        log_b_raw = inputs + info['log_b0']
        log_b = log_normalize(log_b_raw)
        return log_b


class BPGNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=0, dropout_p=0.0, learn_coupling=False):
        super(BPGNN, self).__init__()
        self.transform = nn.Sequential(MLP(dim_in, dim_out, dim_hidden, num_hidden, nn.ReLU(), dropout_p), nn.LogSoftmax(dim=-1))
        self.conv = BPConv(dim_out, learn_coupling)

    def forward(self, x, edge_index, rv, phi=None):
        log_b0 = self.transform(x)
        if phi is not None:
            log_b0[phi[0]] += phi[1]
        num_classes = log_b0.shape[-1]
        info = {'log_b0': log_b0,
                'log_msg_': (-np.log(num_classes)) * torch.ones(edge_index.shape[1], num_classes).to(x.device),
                'rv': rv}
        log_b = log_b0
        for _ in range(5):
            log_b = self.conv(log_b, edge_index, info)

        return log_b


def cts_coupling(edge_index, y):
    idx, cts = torch.stack((y[edge_index[0]], y[edge_index[1]]), dim=0).unique(dim=1, return_counts=True)
    C = torch.zeros(y.max()+1, y.max()+1, device=y.device)
    C[idx[0], idx[1]] = cts.float()
    d = C.sum(dim=0)
    coupling = (d**-0.5).view(-1,1) * C * (d**-0.5).view(1,-1)
    return coupling


def run(dataset, split, model, num_hidden, device, learning_rate, develop):

    if dataset == 'Cora':
        data = load_citation('Cora', split=split)
    elif dataset == 'CiteSeer':
        data = load_citation('CiteSeer', split=split)
    elif dataset == 'PubMed':
        data = load_citation('PubMed', split=split)
    elif dataset == 'Coauthor_CS':
        data = load_coauthor('CS', split=split)
    elif dataset == 'Coauthor_Physics':
        data = load_coauthor('Physics', split=split)
    elif dataset == 'County_Facebook':
        data = load_county_facebook(split=split)
    elif dataset == 'OGBN_Products':
        data = load_ogbn('products', split=split)
    else:
        raise Exception('unexpected dataset')
    data = data.to(device)

    edge_index, rv = data.edge_index, data.rv
    x, y = data.x, data.y
    num_nodes, num_features = x.shape
    num_classes = len(torch.unique(y))
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    if model == 'SGC':
        gnn = SGC(num_features, num_classes)
    elif model == 'GCN':
        gnn = GCN(num_features, num_classes, 128, 0.3)
    elif model == 'GAT':
        gnn = GAT(num_features, num_classes, 32, 0.6)
    elif model == 'BPGNN':
        gnn = BPGNN(num_features, num_classes, 128, num_hidden, 0.3, True)
    else:
        raise Exception('unexpected model')
    gnn = gnn.to(device)

    optimizer = torch.optim.AdamW([{'params': gnn.parameters(), 'lr': learning_rate}], weight_decay=2.5e-4)

    def train():
        gnn.train()
        optimizer.zero_grad()
        b = gnn(x, edge_index, rv=rv)
        loss = F.nll_loss(b[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        if develop:
            print('step {:5d}, train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(epoch, acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)
        return acc(b, y, val_mask)

    def evaluation():
        gnn.eval()
        if type(gnn) == BPGNN:
            sum_conv = SumConv()
            log_e0 = torch.zeros(num_nodes, num_classes).to(device)
            log_e0[train_mask] = gnn.conv.get_logH()[y[train_mask]]
            subgraph_mask = torch.logical_not(train_mask)
            subgraph_edge_index, subgraph_rv = process_edge_index(num_nodes, subgraph(subgraph_mask, edge_index)[0])
            b = gnn(x, subgraph_edge_index, rv=subgraph_rv, phi=(subgraph_mask, sum_conv(log_e0, edge_index)[subgraph_mask]))
            if develop:
                print('train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)
                print(gnn.conv.get_logH().exp())
        else:
            b = gnn(x, edge_index, rv=rv)
            if develop:
                print('evaluation, train accuracy: {:5.3f}, val accuracy: {:5.3f}, test accuracy: {:5.3f}'.format(acc(b, y, train_mask), acc(b, y, val_mask), acc(b, y, test_mask)), flush=True)
        return acc(b, y, val_mask), acc(b, y, test_mask)

    best_val, opt_val, opt_test = 0.0, 0.0, 0.0
    for epoch in range(300):
        val = train()
        if best_val < val:
            best_val = val
            opt_val, opt_test = evaluation()

    if develop:
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))

    return opt_test


parser = argparse.ArgumentParser('gnn')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', metavar='N', type=float, nargs=3, default=None)
parser.add_argument('--model', type=str, default='BPGNN')
parser.add_argument('--num_hidden', type=int, default=2)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--develop', type=bool, default=False)
args = parser.parse_args()

outpath = create_outpath(args.dataset)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.develop:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

test_acc = []
for _ in range(100):
    test_acc.append(run(args.dataset, args.split, args.model, args.num_hidden, args.device, args.learning_rate, args.develop))

print(args)
print('test accuracies: {:7.3f} ± {:7.3f}'.format(np.mean(test_acc)*100, np.std(test_acc)*100))
