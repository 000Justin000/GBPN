import sys
import math
import random
import subprocess
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, is_undirected, subgraph
from sklearn.metrics import roc_auc_score
from utils import *
import matplotlib
import matplotlib.pyplot as plt


def get_scaling(deg0, deg1):
    assert deg0.shape == deg1.shape
    scaling = torch.ones(deg0.shape[0]).to(deg0.device)
#   scaling[deg1 != 0] = (deg0 / deg1)[deg1 != 0]
    return scaling


def get_cts(edge_index, y):
    idx, cts = torch.stack((y[edge_index[0]], y[edge_index[1]]), dim=0).unique(dim=1, return_counts=True)
    ctsm = torch.zeros(y.max() + 1, y.max() + 1, device=y.device)
    ctsm[idx[0], idx[1]] = cts.float()
    sqrt_deg_inv = ctsm.sum(dim=1) ** -0.5
    return ctsm, sqrt_deg_inv.view(-1, 1) * ctsm * sqrt_deg_inv.view(1, -1)


def classification_accuracy(log_b, y):
    if type(log_b) == torch.Tensor:
        log_b = log_b.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    return (log_b.argmax(-1) == y).sum() / y.shape[0]


def roc_auc(log_b, y):
    if type(log_b) == torch.Tensor:
        log_b = log_b.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    return roc_auc_score(y, np.exp(log_b[:, 1]))


def run(dataset, homo_ratio, split, model_name, dim_hidden, num_hidden, dropout_p, device, learning_rate, num_epoches, weighted_BP, learn_H, eval_C, verbose):
    if dataset == 'Cora':
        data = load_citation('Cora', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'CiteSeer':
        data = load_citation('CiteSeer', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'PubMed':
        data = load_citation('PubMed', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Coauthor_CS':
        data = load_coauthor('CS', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Coauthor_Physics':
        data = load_coauthor('Physics', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'County_Facebook':
        data = load_county_facebook(split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Sex':
        data = load_sexual_interaction(split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Animal2':
        data = load_animal2(homo_ratio=homo_ratio, split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Animal3':
        data = load_animal3(homo_ratio=homo_ratio, split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Squirrel':
        data = load_wikipedia('Squirrel', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Chameleon':
        data = load_wikipedia('Chameleon', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'OGBN_arXiv':
        data = load_ogbn('arxiv', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'OGBN_Products':
        data = load_ogbn('products', split=split)
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset == 'Elliptic_Bitcoin':
        data = load_elliptic_bitcoin(split=split)
        _, cts = data.y[data.y >= 0].unique(return_counts=True)
        c_weight = (cts**-1.0) / (cts**-1.0).sum()
        accuracy_fun = roc_auc
    elif dataset == 'JPMC_Fraud_Detection':
        x, y, info = load_jpmc_fraud()
        data = preprocess_gnn_jpmc_fraud(x, y, info, split=split)
        _, cts = data.y.unique(return_counts=True)
        c_weight = (cts**-1.0) / (cts**-1.0).sum()
        accuracy_fun = roc_auc
    else:
        raise Exception('unexpected dataset')

    edge_index, edge_weight, rv = data.edge_index, data.edge_weight, data.rv
    x, y = data.x, data.y
    num_nodes, num_features = x.shape
    num_classes = len(torch.unique(y[y >= 0]))
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    c_weight = None if (c_weight is None) else c_weight.to(device)

    if (model_name == 'GBPN') and weighted_BP and (edge_weight is None):
        deg = degree(edge_index[1], num_nodes)
        edge_weight = (deg[edge_index[0]] * deg[edge_index[1]])**-0.50 * deg.mean()

    if dataset in ['Cora', 'CiteSeer', 'PubMed', 'Coauthor_CS', 'Coauthor_Physics', 'County_Facebook', 'Sex', 'Animal2', 'Animal3', 'Squirrel', 'Chameleon']:
        subgraph_sampler = SubgraphSampler(num_nodes, x, y, edge_index, edge_weight)
        max_batch_size = num_nodes
        if model_name == 'MLP':
            num_hops = 0
        elif model_name != 'GBPN':
            num_hops = 2
        else:
            num_hops = 5
        num_samples = -1
    elif dataset in ['OGBN_arXiv', 'OGBN_Products', 'JPMC_Fraud_Detection', 'Elliptic_Bitcoin']:
        subgraph_sampler = CSubtreeSampler(num_nodes, x, y, edge_index, edge_weight)
        max_batch_size = min(math.ceil(train_mask.sum()/10.0), 1024)
        if model_name == 'MLP':
            num_hops = 0
        else:
            num_hops = 2
        num_samples = 5
    else:
        raise Exception('unexpected dataset encountered')

    if model_name == 'MLP':
        model = GMLP(num_features, num_classes, dim_hidden=dim_hidden, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=dropout_p)
    elif model_name == 'SGC':
        model = SGC(num_features, num_classes, dim_hidden=dim_hidden, dropout_p=dropout_p)
    elif model_name == 'GCN':
        model = GCN(num_features, num_classes, dim_hidden=dim_hidden, activation=nn.LeakyReLU(), dropout_p=dropout_p)
    elif model_name == 'SAGE':
        model = SAGE(num_features, num_classes, dim_hidden=dim_hidden, activation=nn.LeakyReLU(), dropout_p=dropout_p)
    elif model_name == 'GAT':
        model = GAT(num_features, num_classes, dim_hidden=dim_hidden//8, activation=nn.ELU(), dropout_p=dropout_p)
    elif model_name == 'GBPN':
        model = GBPN(num_features, num_classes, dim_hidden=dim_hidden, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=dropout_p, learn_H=learn_H)
    else:
        raise Exception('unexpected model type')
    model = model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2.5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(num_hops=2, num_samples=5):
        model.train()
        total_loss = 0.0
        log_b_list, gth_y_list = [], []
        for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(train_mask, max_batch_size, num_hops, num_samples, device):
            optimizer.zero_grad()
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, agg_scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), rv=subgraph_rv, K=num_hops)
            loss = F.nll_loss(subgraph_log_b[:batch_size], batch_y, weight=c_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)*batch_size
            log_b_list.append(subgraph_log_b[:batch_size].detach())
            gth_y_list.append(batch_y)
        mean_loss = total_loss / int(train_mask.sum())
        accuracy = accuracy_fun(torch.cat(log_b_list, dim=0), torch.cat(gth_y_list, dim=0))
        if verbose:
            print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, mean_loss, accuracy), end='    ')
        return accuracy

    def evaluation(mask, num_hops=2, num_samples=5, partition='train'):
        model.eval()
        log_b_list, gth_y_list = [], []
        for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, num_samples, device):
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, agg_scaling=get_scaling(subgraph_deg0, degree(subgraph_edge_index[1], subgraph_size)), rv=subgraph_rv, K=num_hops)
            log_b_list.append(subgraph_log_b[:batch_size].detach())
            gth_y_list.append(batch_y)
        accuracy = accuracy_fun(torch.cat(log_b_list, dim=0), torch.cat(gth_y_list, dim=0))
        if verbose:
            print('{:>5s} inductive accuracy: {:5.3f}'.format(partition, accuracy), end='    ')
        if type(model) == GBPN and eval_C:
            sum_conv = SumConv()
            log_b_list, gth_y_list = [], []
            for batch_size, batch_nodes, batch_x, batch_y, batch_deg0, \
                subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg0, \
                subgraph_edge_index, subgraph_edge_weight, subgraph_rv in subgraph_sampler.get_generator(mask, max_batch_size, num_hops, num_samples, device):
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
                log_b_list.append(subgraph_log_b[:batch_size].detach())
                gth_y_list.append(batch_y)
            accuracy = accuracy_fun(torch.cat(log_b_list, dim=0), torch.cat(gth_y_list, dim=0))
            if verbose:
                print('{:>5s} transductive accuracy: {:5.3f}'.format(partition, accuracy), end='    ')
        return accuracy

    max_num_hops = num_hops
    best_val, opt_val, opt_test = 0.0, 0.0, 0.0
    for epoch in range(num_epoches):
        num_hops = 0 if (model_name == 'GBPN' and epoch < 0.05*num_epoches) else max_num_hops
        train(num_hops=num_hops, num_samples=num_samples)

        with torch.no_grad():
            val = evaluation(val_mask, num_hops=num_hops, num_samples=num_samples, partition='val')
            if val > opt_val:
                opt_val = val
                opt_test = evaluation(test_mask, num_hops=num_hops, num_samples=num_samples, partition='test')
        print(flush=True)

        if type(model) == GBPN and learn_H and verbose:
            print(model.bp_conv.get_logH().exp(), flush=True)

    if verbose:
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))

    return opt_test


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_printoptions(precision=4, threshold=None, edgeitems=5, linewidth=300, profile=None, sci_mode=False)
parser = argparse.ArgumentParser('model')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--homo_ratio', type=float, default=0.5)
parser.add_argument('--split', metavar='N', type=float, nargs=3, default=None)
parser.add_argument('--model_name', type=str, default='GBPN')
parser.add_argument('--dim_hidden', type=int, default=256)
parser.add_argument('--num_hidden', type=int, default=2)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--num_epoches', type=int, default=20)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--weighted_BP', action='store_true')
parser.add_argument('--learn_H', action='store_true')
parser.add_argument('--eval_C', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--develop', action='store_true')
parser.set_defaults(weighted_BP=False, learn_H=False, eval_C=False, verbose=False, develop=False)
args = parser.parse_args()

outpath = create_outpath(args.dataset, args.model_name)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.develop:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

test_acc = []
for _ in range(args.num_trials):
    test_acc.append(run(args.dataset, args.homo_ratio, args.split, args.model_name, args.dim_hidden, args.num_hidden, args.dropout_p, args.device, args.learning_rate, args.num_epoches, args.weighted_BP, args.learn_H, args.eval_C, args.verbose))

print(args)
print('overall test accuracies: {:7.3f} ± {:7.3f}'.format(np.mean(test_acc) * 100, np.std(test_acc) * 100))

print('finish')
