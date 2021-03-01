import sys
import math
import subprocess
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, sort_edge_index, is_undirected, subgraph, k_hop_subgraph
from utils import *
import matplotlib
import matplotlib.pyplot as plt


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
        logT[rid, cid] = F.logsigmoid(self.param)
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
        log_b_raw = info['log_b0'] + agg_log_msg + (info['scaling']-1.0).unsqueeze(-1) * agg_log_msg.detach()
        log_b = log_normalize(log_b_raw)
        return log_b


class BPGNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=0, activation=nn.ReLU(), dropout_p=0.0, learn_H=False):
        super(BPGNN, self).__init__()
        self.transform = nn.Sequential(MLP(dim_in, dim_out, dim_hidden, num_hidden, activation, dropout_p), nn.LogSoftmax(dim=-1))
        self.conv = BPConv(dim_out, learn_H)

    def forward(self, x, edge_index, edge_weight=None, rv=None, phi=None, scaling=None, K=5):
        log_b0 = self.transform(x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        if rv is None:
            edge_index, edge_weight, rv = process_edge_index(x.shape[0], edge_index, edge_weight)
        if phi is not None:
            log_b0 += phi
        if scaling is None:
            scaling = torch.ones(x.shape[0]).to(x.device)
        num_classes = log_b0.shape[-1]
        info = {'log_b0': log_b0,
                'log_msg_': (-np.log(num_classes)) * torch.ones(edge_index.shape[1], num_classes).to(x.device),
                'rv': rv,
                'scaling': scaling}
        log_b = log_b0
        for _ in range(K):
            log_b = self.conv(log_b, edge_index, edge_weight, info)
        return log_b


def get_cts(edge_index, y):
    idx, cts = torch.stack((y[edge_index[0]], y[edge_index[1]]), dim=0).unique(dim=1, return_counts=True)
    ctsm = torch.zeros(y.max() + 1, y.max() + 1, device=y.device)
    ctsm[idx[0], idx[1]] = cts.float()
    sqrt_deg_inv = ctsm.sum(dim=1) ** -0.5
    return ctsm, sqrt_deg_inv.view(-1, 1) * ctsm * sqrt_deg_inv.view(1, -1)


def run(dataset, homo_ratio, split, model_name, num_hidden, device, learning_rate, train_BP, learn_H, eval_C, develop):
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
    elif dataset == 'Sex':
        data = load_sexual_interaction(split=split)
    elif dataset == 'Cats_Dogs':
        data = load_cats_dogs(homo_ratio=homo_ratio, split=split)
    elif dataset == 'Animals':
        data = load_animals(homo_ratio=homo_ratio, split=split)
    elif dataset == 'Squirrel':
        data = load_wikipedia('Squirrel', split=split)
    elif dataset == 'Chameleon':
        data = load_wikipedia('Chameleon', split=split)
    elif dataset == 'OGBN_Products':
        data = load_ogbn('products', split=split)
    else:
        raise Exception('unexpected dataset')

    edge_index, edge_weight, rv = data.edge_index, data.edge_weight, data.rv
    x, y = data.x, data.y
    num_nodes, num_features = x.shape
    num_classes = len(torch.unique(y))
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    subgraph_sampler = SubgraphSampler(num_nodes, x, y, edge_index, edge_weight)
    max_batch_size = 128

    if model_name == 'MLP':
        model = GMLP(num_features, num_classes, dim_hidden=128, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=0.3)
    elif model_name == 'SGC':
        model = SGC(num_features, num_classes, dim_hidden=128, dropout_p=0.3)
    elif model_name == 'GCN':
        model = GCN(num_features, num_classes, dim_hidden=128, activation=nn.LeakyReLU(), dropout_p=0.3)
    elif model_name == 'SAGE':
        model = SAGE(num_features, num_classes, dim_hidden=128, activation=nn.LeakyReLU(), dropout_p=0.3)
    elif model_name == 'GAT':
        model = GAT(num_features, num_classes, dim_hidden=8, activation=nn.ELU(), dropout_p=0.6)
    elif model_name == 'BPGNN':
        model = BPGNN(num_features, num_classes, dim_hidden=128, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=0.3, learn_H=learn_H)
    else:
        raise Exception('unexpected model type')
    model = model.to(device)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': learning_rate}], weight_decay=2.5e-4)

    def train(num_hops=3, max_neighbors=5):
        model.train()
        n_batch = total_loss = total_correct = 0.0
        for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(train_mask, max_batch_size, num_hops, max_neighbors, device):
            optimizer.zero_grad()
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, rv=subgraph_rv, scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), K=num_hops)
            loss = F.nll_loss(subgraph_log_b[:batch_size], batch_y)
            loss.backward()
            optimizer.step()
            n_batch += 1
            total_loss += float(loss)
            total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()

        if develop:
            print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, total_loss/n_batch, total_correct/train_mask.sum().item()), flush=True)

        return total_correct/train_mask.sum().item()

    def evaluation(mask, num_hops=3, max_neighbors=5):
        model.eval()
        if type(model) == BPGNN and develop:
            print(model.conv.get_logH().exp())
        total_correct = 0.0
        for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, max_neighbors, device):
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, rv=subgraph_rv, scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), K=num_hops)
            total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()
        if develop:
            print('accuracy: {:5.3f}'.format(total_correct/mask.sum().item()), flush=True)

        if type(model) == BPGNN and eval_C:
            sum_conv = SumConv()
            total_correct = 0.0
            for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
                subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, max_neighbors, device):
                subgraphC_mask = train_mask[subgraph_nodes]
                subgraphR_mask = torch.logical_not(subgraphC_mask)
                log_c = torch.zeros(subgraph_size, num_classes).to(device)
                log_c[subgraphC_mask] = model.conv.get_logH()[subgraph_y[subgraphC_mask]]
                subgraphR_phi = sum_conv(log_c, subgraph_edge_index, subgraph_edge_weight)[subgraphR_mask]
                subgraphR_edge_index, subgraphR_edge_weight = subgraph(subgraphR_mask, subgraph_edge_index, subgraph_edge_weight, relabel_nodes=True)
                subgraphR_edge_index, subgraphR_edge_weight, subgraphR_rv = process_edge_index(subgraphR_mask.sum().item(), subgraphR_edge_index, subgraphR_edge_weight)
                subgraph_log_b = torch.zeros(subgraph_size, num_classes).to(device)
                subgraph_log_b[subgraphC_mask] = F.one_hot(subgraph_y[subgraphC_mask], num_classes).float().to(device)
                subgraph_log_b[subgraphR_mask] = model(subgraph_x[subgraphR_mask], subgraphR_edge_index, subgraphR_edge_weight, rv=subgraphR_rv, phi=subgraphR_phi,
                                                       scaling=get_scaling(subgraph_deg0[subgraphR_mask], degree(subgraphR_edge_index[1], subgraphR_mask.sum().item())), K=num_hops)
                total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()
            if develop:
                print('accuracy: {:5.3f}'.format(total_correct/mask.sum().item()), flush=True)

            return total_correct/mask.sum().item()

    best_val, opt_val, opt_test = 0.0, 0.0, 0.0
    for epoch in range(30):
        num_hops = (3 if (epoch >= 3 and train_BP) else 0)
        max_neighbors = 5
        train(num_hops=num_hops, max_neighbors=max_neighbors)
        val = evaluation(val_mask, num_hops=num_hops, max_neighbors=max_neighbors)
        if val > opt_val:
            opt_val = val
            opt_test = evaluation(test_mask, num_hops=num_hops, max_neighbors=max_neighbors)

    if develop:
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))

    return opt_test


torch.set_printoptions(precision=4, threshold=None, edgeitems=15, linewidth=200, profile=None, sci_mode=False)
parser = argparse.ArgumentParser('model')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--homo_ratio', type=float, default=0.5)
parser.add_argument('--split', metavar='N', type=float, nargs=3, default=None)
parser.add_argument('--model_name', type=str, default='BPGNN')
parser.add_argument('--num_hidden', type=int, default=2)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--train_BP', action='store_true')
parser.add_argument('--learn_H', action='store_true')
parser.add_argument('--eval_C', action='store_true')
parser.add_argument('--develop', action='store_true')
parser.set_defaults(train_BP=False, learn_H=False, eval_C=False, develop=False)
args = parser.parse_args()

outpath = create_outpath(args.dataset, args.model_name)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.develop:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

test_acc = []
for _ in range(30):
    test_acc.append(run(args.dataset, args.homo_ratio, args.split, args.model_name, args.num_hidden, args.device, args.learning_rate, args.train_BP, args.learn_H, args.eval_C, args.develop))

print(args)
print('overall test accuracies: {:7.3f} ± {:7.3f}'.format(np.mean(test_acc) * 100, np.std(test_acc) * 100))
