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
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import degree, is_undirected, subgraph
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from utils import *
import matplotlib
import matplotlib.pyplot as plt


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


def optimal_f1_score(log_b, y, optimal_threshold=None):
    if type(log_b) == torch.Tensor:
        log_b = log_b.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    y_probs = np.exp(log_b[:, 1])
    if optimal_threshold == None:
        precisions, recalls, thresholds = precision_recall_curve(y, y_probs)
        f1_scores = 2*recalls*precisions/(recalls+precisions+sys.float_info.epsilon)
        return np.max(f1_scores), thresholds[np.argmax(f1_scores)]
    else:
        y_preds = (y_probs > optimal_threshold).astype(int)
        return f1_score(y, y_preds)


def run(dataset, split, model_name, dim_hidden, num_layers, num_hops, num_samples, dropout_p, device, learning_rate, num_epoches, loss_option, weighted_BP, deg_scaling, learn_H, eval_C, verbose):
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
        data.y[data.y < 0] = 0
        c_weight = (cts**-1.0) / (cts**-1.0).sum()
        accuracy_fun = optimal_f1_score
    elif dataset in ['JPMC_Payment0', 'JPMC_Payment1']:
        x, y, info = load_jpmc_payment(dataset[-1])
        data = preprocess_gnn_jpmc_payment(x, y, info, split=split)
        _, cts = data.y.unique(return_counts=True)
        c_weight = (cts**-1.0) / (cts**-1.0).sum()
        accuracy_fun = optimal_f1_score
    elif dataset in ['Ising+', 'Ising-']:
        data = load_ising(split=split, interaction=dataset[-1], dataset_id=np.random.randint(10))
        c_weight = None
        accuracy_fun = classification_accuracy
    elif dataset in ['MRF+', 'MRF-']:
        data = load_mrf(split=split, interaction=dataset[-1], dataset_id=np.random.randint(10))
        c_weight = None
        accuracy_fun = classification_accuracy
    else:
        raise Exception('unexpected dataset')

    edge_index, edge_weight, edge_rv = data.edge_index, data.edge_weight, data.edge_rv
    x, y = data.x, data.y
    num_nodes, num_features = x.shape
    num_classes = len(torch.unique(y[y >= 0]))
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    deg = degree(edge_index[1], num_nodes)
    c_weight = None if (c_weight is None) else c_weight.to(device)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    if (model_name == 'GBPN') and weighted_BP:
        edge_weight = ((deg[edge_index[0]] * deg[edge_index[1]])**-0.5 * deg.mean())

    if dataset in ['Cora', 'CiteSeer', 'PubMed', 'Coauthor_CS', 'Coauthor_Physics', 'County_Facebook', 'Sex', 'Animal2', 'Animal3', 'Squirrel', 'Chameleon', 'Ising+', 'Ising-', 'MRF+', 'MRF-']:
        graph_sampler = FullgraphSampler(num_nodes, x, y, edge_index, edge_weight, edge_rv)
        max_batch_size = -1
    elif dataset in ['OGBN_arXiv', 'OGBN_Products', 'JPMC_Payment0', 'JPMC_Payment1', 'Elliptic_Bitcoin']:
        graph_sampler = SubtreeSampler(num_nodes, x, y, edge_index, edge_weight, edge_rv)
        # graph_sampler = ClusterSampler(num_nodes, x, y, edge_index, edge_weight, edge_rv, train_mask, val_mask, test_mask)
        max_batch_size = min(math.ceil(train_mask.sum()/10.0), 512)
    else:
        raise Exception('unexpected dataset encountered')

    if model_name == 'LP':
        row, col = edge_index
        perm = (col * num_nodes + row).argsort()
        row, col = row[perm], col[perm]
        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
        model = LabelPropagation(num_layers=50, alpha=0.00)
    elif model_name == 'MLP':
        model = GMLP(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, activation=nn.ReLU(), dropout_p=dropout_p)
    elif model_name == 'SGC':
        model = SGC(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, dropout_p=dropout_p)
    elif model_name == 'GCN':
        model = GCN(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, activation=nn.ReLU(), dropout_p=dropout_p)
    elif model_name == 'SAGE':
        model = SAGE(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, activation=nn.ReLU(), dropout_p=dropout_p)
    elif model_name == 'GAT':
        model = GAT(num_features, num_classes, dim_hidden=dim_hidden//4, num_layers=num_layers, num_heads=4, activation=nn.ELU(), dropout_p=dropout_p)
    elif model_name == 'GBPN':
        model = GBPN(num_features, num_classes, dim_hidden=dim_hidden, num_layers=num_layers, activation=nn.ReLU(), dropout_p=dropout_p, loss_option=loss_option, deg_scaling=deg_scaling, learn_H=learn_H)
    else:
        raise Exception('unexpected model type')
    model = model.to(device)

    def loss_and_accuracy(y, log_b):
        train_loss = F.nll_loss(log_b[train_mask], y[train_mask], weight=c_weight)
        val_loss = F.nll_loss(log_b[val_mask], y[val_mask], weight=c_weight)
        test_loss = F.nll_loss(log_b[test_mask], y[test_mask], weight=c_weight)
        if accuracy_fun == optimal_f1_score:
            train_accuracy, _ = optimal_f1_score(log_b[train_mask], y[train_mask])
            val_accuracy, optimal_threshold = optimal_f1_score(log_b[val_mask], y[val_mask])
            test_accuracy = optimal_f1_score(log_b[test_mask], y[test_mask], optimal_threshold=optimal_threshold)
        else:
            train_accuracy = accuracy_fun(log_b[train_mask], y[train_mask])
            val_accuracy = accuracy_fun(log_b[val_mask], y[val_mask])
            test_accuracy = accuracy_fun(log_b[test_mask], y[test_mask])
        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy

    def accuracy_degree_correlation(log_b, y, deg):
        nll = log_b.gather(-1, y.reshape(-1,1))
        crs = log_b.argmax(dim=-1) == y
        _, perm = deg.sort()
        deg_avg = [float(batch.float().mean()) for batch in deg[perm].chunk(100)]
        nll_avg = [float(batch.float().mean()) for batch in nll[perm].chunk(100)]
        crs_avg = [float(batch.float().mean()) for batch in crs[perm].chunk(100)]
        return deg_avg, nll_avg, crs_avg

    if model_name == 'LP':
        opt_val, opt_test = 0.0, 0.0
        opt_deg_avg, opt_nll_avg, opt_crs_avg = None, None, None
        for alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
            model.alpha = alpha
            log_b = log_normalize(model(y.to(device), adj_t.to(device), train_mask.to(device), post_step=lambda y: y.clamp_(1.0e-15,1.0e0)).log()).cpu()
            train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = loss_and_accuracy(y.to(device), log_b.to(device))
            deg_avg, nll_avg, crs_avg = accuracy_degree_correlation(log_b[test_mask], y[test_mask], deg[test_mask])
            if val_accuracy > opt_val:
                opt_val = val_accuracy
                opt_test = test_accuracy
                opt_deg_avg = deg_avg
                opt_nll_avg = nll_avg
                opt_crs_avg = crs_avg
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))
        print('optimal deg average: [' + ', '.join(map(lambda f: '{:7.3f}'.format(f), opt_deg_avg)) + ']')
        print('optimal nll average: [' + ', '.join(map(lambda f: '{:7.3f}'.format(f), opt_nll_avg)) + ']')
        print('optimal crs average: [' + ', '.join(map(lambda f: '{:7.3f}'.format(f), opt_crs_avg)) + ']')
        return opt_test, opt_deg_avg, opt_nll_avg, opt_crs_avg
    elif model_name == 'GBPN':
        optimizer = MultiOptimizer(torch.optim.AdamW(model.transform.parameters(), lr=learning_rate, weight_decay=2.5e-4),
                                   torch.optim.AdamW(model.bp_conv.parameters(), lr=learning_rate*10, weight_decay=2.5e-4))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2.5e-4)

    def train(num_hops=2, num_samples=5):
        model.train()
        total_loss = 0.0
        log_b_list, gth_y_list = [], []
        for batch_size, batch_nodes, _, batch_y, _, \
            subgraph_size, subgraph_nodes, subgraph_x, subgraph_y, subgraph_deg, \
            subgraph_edge_index, subgraph_edge_weight, subgraph_edge_rv, _ in graph_sampler.get_generator(mask=train_mask, shuffle=True, max_batch_size=max_batch_size, num_hops=num_hops, num_samples=num_samples, device=device):

            phi = torch.zeros(subgraph_size, num_classes).to(device)
            backpp_mask = torch.ones(batch_size, dtype=torch.bool).to(device)
            if type(model) == GBPN and eval_C:
                backpp_nodes = batch_nodes[torch.rand(batch_size) > 0.5]
                anchor_mask = train_mask.clone()
                anchor_mask[backpp_nodes] = False
                phi[anchor_mask[subgraph_nodes]] = torch.log(F.one_hot(subgraph_y[anchor_mask[subgraph_nodes]], num_classes).float())
                backpp_mask[anchor_mask[batch_nodes]] = False

            optimizer.zero_grad()
            subgraph_log_b = model(subgraph_x, subgraph_edge_index, edge_weight=subgraph_edge_weight, edge_rv=subgraph_edge_rv,
                                   deg=degree(subgraph_edge_index[1], subgraph_size), deg_ori=deg[subgraph_nodes].to(device), phi=phi, K=num_hops)
            loss = F.nll_loss(subgraph_log_b[:batch_size][backpp_mask], batch_y[backpp_mask], weight=c_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)*batch_size
            log_b_list.append(subgraph_log_b[:batch_size][backpp_mask].detach().cpu())
            gth_y_list.append(batch_y[backpp_mask].cpu())
        mean_loss = total_loss / int(train_mask.sum())
        if accuracy_fun == optimal_f1_score:
            accuracy, _ = optimal_f1_score(torch.cat(log_b_list, dim=0), torch.cat(gth_y_list, dim=0))
        else:
            accuracy = accuracy_fun(torch.cat(log_b_list, dim=0), torch.cat(gth_y_list, dim=0))
        if verbose:
            print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, mean_loss, accuracy), end='    ', flush=True)
        return accuracy

    @torch.no_grad()
    def evaluation(num_hops=2):
        model.eval()
        log_b = model.inference(graph_sampler, max_batch_size, device, K=num_hops)
        train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = loss_and_accuracy(y.to(device), log_b.to(device))
        if verbose:
            print('inductive loss / accuracy: ({:5.3f}, {:5.3f}, {:5.3f}) / ({:5.3f}, {:5.3f}, {:5.3f})'.format(train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy), end='    ', flush=True)

        if type(model) == GBPN and eval_C:
            phi = torch.zeros(num_nodes, num_classes)
            phi[train_mask] = torch.log(F.one_hot(y[train_mask], num_classes).float())
            log_b = model.inference(graph_sampler, max_batch_size, device, phi=phi, K=num_hops)
            train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = loss_and_accuracy(y.to(device), log_b.to(device))
            if verbose:
                print('transductive loss / accuracy: ({:5.3f}, {:5.3f}, {:5.3f}) / ({:5.3f}, {:5.3f}, {:5.3f})'.format(train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy), end='', flush=True)

        return train_accuracy, val_accuracy, test_accuracy, log_b

    max_num_hops = num_hops
    opt_val, opt_test = 0.0, 0.0
    opt_deg_avg, opt_nll_avg, opt_crs_avg = None, None, None
    for epoch in range(1, num_epoches+1):
        num_hops = 0 if (model_name == 'GBPN' and epoch <= num_epoches*0.05) else max_num_hops
        train(num_hops=num_hops, num_samples=num_samples)

        if epoch % max(int(num_epoches*0.1), 10) == 0:
            train_accuracy, val_accuracy, test_accuracy, log_b = evaluation(num_hops=num_hops)
            deg_avg, nll_avg, crs_avg = accuracy_degree_correlation(log_b[test_mask], y[test_mask], deg[test_mask])
            print()
            print('deg average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), deg_avg)) + ']')
            print('nll average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), nll_avg)) + ']')
            print('crs average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), crs_avg)) + ']')

            if val_accuracy > opt_val:
                opt_val = val_accuracy
                opt_test = test_accuracy
                opt_deg_avg = deg_avg
                opt_nll_avg = nll_avg
                opt_crs_avg = crs_avg
            if type(model) == GBPN and verbose:
                print(model.bp_conv.get_logH().exp(), flush=True)
        print(flush=True)

    if verbose:
        print('optimal val accuracy: {:7.5f}, optimal test accuracy: {:7.5f}'.format(opt_val, opt_test))
        print('optimal deg average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), opt_deg_avg)) + ']')
        print('optimal nll average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), opt_nll_avg)) + ']')
        print('optimal crs average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), opt_crs_avg)) + ']')
        print()

    return opt_test, opt_deg_avg, opt_nll_avg, opt_crs_avg


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_printoptions(precision=4, threshold=None, edgeitems=10, linewidth=300, profile=None, sci_mode=False)
parser = argparse.ArgumentParser('model')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', metavar='N', type=float, nargs=3, default=None)
parser.add_argument('--model_name', type=str, default='GBPN')
parser.add_argument('--dim_hidden', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--num_epoches', type=int, default=20)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--loss_option', type=int, default=5)
parser.add_argument('--weighted_BP', action='store_true')
parser.add_argument('--deg_scaling', action='store_true')
parser.add_argument('--learn_H', action='store_true')
parser.add_argument('--eval_C', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--develop', action='store_true')
parser.set_defaults(weighted_BP=False, deg_scaling=False, learn_H=False, eval_C=False, verbose=False, develop=False)
args = parser.parse_args()

outpath = create_outpath(args.dataset, args.model_name)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.develop:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

test_acc = []
test_deg_avg = []
test_nll_avg = []
test_crs_avg = []
for _ in range(args.num_trials):
    opt_test, opt_deg_avg, opt_nll_avg, opt_crs_avg = run(args.dataset, args.split, args.model_name, args.dim_hidden, args.num_layers, args.num_hops, args.num_samples, args.dropout_p, args.device, args.learning_rate, args.num_epoches, args.loss_option, args.weighted_BP, args.deg_scaling, args.learn_H, args.eval_C, args.verbose)
    test_acc.append(opt_test)
    test_deg_avg.append(opt_deg_avg)
    test_nll_avg.append(opt_nll_avg)
    test_crs_avg.append(opt_crs_avg)

list_avg = lambda ll: list(map(lambda l: sum(l)/len(l), zip(*ll)))

print(args)
print('overall test accuracies: {:7.3f} ± {:7.3f}'.format(np.mean(test_acc)*100, np.std(test_acc)*100))
print('optimal deg average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), list_avg(test_deg_avg))) + ']')
print('optimal nll average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), list_avg(test_nll_avg))) + ']')
print('optimal crs average: [' + ' '.join(map(lambda f: '{:7.3f}'.format(f), list_avg(test_crs_avg))) + ']')
