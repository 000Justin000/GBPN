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
from torch_geometric.utils import degree, is_undirected, subgraph
from utils import *
import matplotlib
import matplotlib.pyplot as plt


def get_scaling(deg0, deg1):
    assert deg0.shape == deg1.shape
    scaling = torch.ones(deg0.shape[0]).to(deg0.device)
    # scaling[deg1 != 0] = (deg0 / deg1)[deg1 != 0]
    return scaling


def get_cts(edge_index, y):
    idx, cts = torch.stack((y[edge_index[0]], y[edge_index[1]]), dim=0).unique(dim=1, return_counts=True)
    ctsm = torch.zeros(y.max() + 1, y.max() + 1, device=y.device)
    ctsm[idx[0], idx[1]] = cts.float()
    sqrt_deg_inv = ctsm.sum(dim=1) ** -0.5
    return ctsm, sqrt_deg_inv.view(-1, 1) * ctsm * sqrt_deg_inv.view(1, -1)


def run(dataset, homo_ratio, split, model_name, num_hidden, device, learning_rate, train_BP, learn_H, eval_C, verbose):
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
    elif dataset == 'OGBN_arXiv':
        data = load_ogbn('arxiv', split=split)
    elif dataset == 'OGBN_Products':
        data = load_ogbn('products', split=split)
    else:
        raise Exception('unexpected dataset')

    edge_index, edge_weight, rv = data.edge_index, data.edge_weight, data.rv
    x, y = data.x, data.y
    num_nodes, num_features = x.shape
    num_classes = len(torch.unique(y))
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    subgraph_sampler = CSubtreeSampler(num_nodes, x, y, edge_index, edge_weight)
    max_batch_size = min(math.ceil(num_nodes/10), 1536)

    if model_name == 'MLP':
        model = GMLP(num_features, num_classes, dim_hidden=128, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=0.1)
    elif model_name == 'SGC':
        model = SGC(num_features, num_classes, dim_hidden=128, dropout_p=0.3)
    elif model_name == 'GCN':
        model = GCN(num_features, num_classes, dim_hidden=128, activation=nn.LeakyReLU(), dropout_p=0.3)
    elif model_name == 'SAGE':
        model = SAGE(num_features, num_classes, dim_hidden=128, activation=nn.LeakyReLU(), dropout_p=0.3)
    elif model_name == 'GAT':
        model = GAT(num_features, num_classes, dim_hidden=8, activation=nn.ELU(), dropout_p=0.6)
    elif model_name == 'BPGNN':
        model = BPGNN(num_features, num_classes, dim_hidden=128, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=0.1, nbr_connection=False, learn_H=learn_H)
    else:
        raise Exception('unexpected model type')
    model = model.to(device)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': learning_rate}], weight_decay=2.5e-4)


    def train(num_hops=2, num_nbrs=5):
        model.train()
        n_batch = total_loss = total_correct = 0.0
        for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(train_mask, max_batch_size, num_hops, num_nbrs, device):
            optimizer.zero_grad()
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, agg_scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), rv=subgraph_rv, K=num_hops)
            loss = F.nll_loss(subgraph_log_b[:batch_size], batch_y)
            loss.backward()
            optimizer.step()
            n_batch += 1
            total_loss += float(loss)
            total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()

        if verbose:
            print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, total_loss/n_batch, total_correct/train_mask.sum().item()), flush=True)

        return total_correct/train_mask.sum().item()


    def evaluation(mask, num_hops=2, num_nbrs=5, partition='train'):
        model.eval()

        total_correct = 0.0
        with torch.no_grad():
            for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
                subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, num_nbrs, device):
                subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, agg_scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), rv=subgraph_rv, K=num_hops)
                total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()
        if verbose:
            print('{:>5s} inductive accuracy: {:5.3f}'.format(partition, total_correct/mask.sum().item()), flush=True)

        if type(model) == BPGNN and eval_C:
            sum_conv = SumConv()
            total_correct = 0.0
            with torch.no_grad():
                for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
                    subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
                    subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, num_nbrs, device):
                    subgraphC_mask = train_mask[subgraph_nodes]
                    subgraphR_mask = torch.logical_not(subgraphC_mask)
                    log_c = torch.zeros(subgraph_size, num_classes).to(device)
                    log_c[subgraphC_mask] = model.bp_conv.get_logH()[subgraph_y[subgraphC_mask]]
                    subgraphR_phi = sum_conv(log_c, subgraph_edge_index, subgraph_edge_weight)[subgraphR_mask]
                    subgraphR_edge_index, subgraphR_edge_weight = subgraph(subgraphR_mask, subgraph_edge_index, subgraph_edge_weight, relabel_nodes=True)
                    subgraphR_edge_index, subgraphR_edge_weight, subgraphR_rv = process_edge_index(subgraphR_mask.sum().item(), subgraphR_edge_index, subgraphR_edge_weight)
                    subgraph_log_b = torch.zeros(subgraph_size, num_classes).to(device)
                    subgraph_log_b[subgraphC_mask] = F.one_hot(subgraph_y[subgraphC_mask], num_classes).float().to(device)
                    subgraph_log_b[subgraphR_mask] = model(subgraph_x[subgraphR_mask], subgraphR_edge_index, subgraphR_edge_weight,
                                                           agg_scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size))[subgraphR_mask], rv=subgraphR_rv, phi=subgraphR_phi, K=num_hops)
                    total_correct += (subgraph_log_b[:batch_size].argmax(-1) == batch_y).sum().item()
            if verbose:
                print('{:>5s} transductive accuracy: {:5.3f}'.format(partition, total_correct/mask.sum().item()), flush=True)

        return total_correct/mask.sum().item()


    best_val, opt_val, opt_test = 0.0, 0.0, 0.0
    for epoch in range(30):
        num_hops = (0 if ((not train_BP) or (learn_H and epoch < 1)) else 3)
        num_nbrs = 5
        train(num_hops=num_hops, num_nbrs=num_nbrs)
        val = evaluation(val_mask, num_hops=num_hops, num_nbrs=num_nbrs, partition='val')
        if val > opt_val:
            opt_val = val
            opt_test = evaluation(test_mask, num_hops=num_hops, num_nbrs=num_nbrs, partition='test')

        if type(model) == BPGNN and learn_H and verbose:
            print(model.bp_conv.get_logH().exp())

    if verbose:
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))

    return opt_test


torch.set_printoptions(precision=4, threshold=None, edgeitems=5, linewidth=200, profile=None, sci_mode=False)
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
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--develop', action='store_true')
parser.set_defaults(train_BP=False, learn_H=False, eval_C=False, verbose=False, develop=False)
args = parser.parse_args()

outpath = create_outpath(args.dataset, args.model_name)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.develop:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

test_acc = []
for _ in range(5):
    test_acc.append(run(args.dataset, args.homo_ratio, args.split, args.model_name, args.num_hidden, args.device, args.learning_rate, args.train_BP, args.learn_H, args.eval_C, args.verbose))

print(args)
print('overall test accuracies: {:7.3f} ± {:7.3f}'.format(np.mean(test_acc) * 100, np.std(test_acc) * 100))

print('finish')
