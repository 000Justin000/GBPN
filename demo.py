import sys
import math
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


def classification_accuracy(log_b, y):
    if type(log_b) == torch.Tensor:
        log_b = log_b.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    return (log_b.argmax(-1) == y).sum() / y.shape[0]


split = [0.3, 0.2, 0.5]
num_hidden = 2
device = 'cuda'
learning_rate = 0.01
num_epoches = 200
learn_H = True
eval_C = False

# data = load_county_facebook(split=split)
data = load_citation('Cora', split=split)
# data = load_citation('PubMed', split=split)

edge_index, edge_weight, rv = data.edge_index, data.edge_weight, data.rv
x, y = data.x, data.y
num_nodes, num_features = x.shape
num_classes = len(torch.unique(y[y >= 0]))
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

c_weight = None
accuracy_fun = classification_accuracy

max_batch_size = num_nodes
max_num_hops = 5

model = GBPN(num_features, num_classes, dim_hidden=256, num_hidden=num_hidden, activation=nn.LeakyReLU(), dropout_p=0.1, learn_H=learn_H)
# model = GCN(num_features, num_classes, dim_hidden=256, activation=nn.LeakyReLU(), dropout_p=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2.5e-4)


def train(epoch, num_hops=2):
    model.train()
    optimizer.zero_grad()
    log_b = model(x, edge_index, edge_weight=edge_weight, rv=rv, K=num_hops)
    loss = F.nll_loss(log_b[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    accuracy = accuracy_fun(log_b[train_mask], y[train_mask])
    print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, float(loss), accuracy), end='    ')
    return accuracy


def evaluation(mask, num_hops=2, partition='train'):
    model.eval()
    log_b = model(x, edge_index, edge_weight=edge_weight, rv=rv, K=num_hops)
    accuracy = accuracy_fun(log_b[mask], y[mask])
    print('{:>5s} inductive accuracy: {:5.3f}'.format(partition, accuracy), end='    ')

    if type(model) == GBPN and eval_C:
        sum_conv = SumConv()
        subgraphC_mask = train_mask
        subgraphR_mask = torch.logical_not(subgraphC_mask)
        log_c = torch.zeros(num_nodes, num_classes)
        log_c[subgraphC_mask] = model.bp_conv.get_logH()[y[subgraphC_mask]]
        subgraphR_phi = sum_conv(log_c, edge_index, edge_weight)[subgraphR_mask]
        subgraphR_edge_index, subgraphR_edge_weight = subgraph(subgraphR_mask, edge_index, edge_weight, relabel_nodes=True)
        subgraphR_edge_index, subgraphR_edge_weight, subgraphR_rv = process_edge_index(subgraphR_mask.sum().item(), subgraphR_edge_index, subgraphR_edge_weight)
        subgraphR_log_b = model(x[subgraphR_mask], subgraphR_edge_index, subgraphR_edge_weight, rv=subgraphR_rv, phi=subgraphR_phi, K=num_hops)
        accuracy = accuracy_fun(subgraphR_log_b[mask[subgraphR_mask]], y[mask])
        print('{:>5s} transductive accuracy: {:5.3f}'.format(partition, accuracy), end='    ')

    return accuracy


optimal_val_accuracy = 0.0
for epoch in range(100):
    num_hops = 0 if (epoch < num_epoches*0.05) else max_num_hops
    train(epoch, num_hops)
    val_accuracy = evaluation(val_mask, num_hops, 'val')
    print(flush=True)
    if type(model) == GBPN:
        print(model.bp_conv.get_logH().exp())
    if val_accuracy > optimal_val_accuracy:
        optimal_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))

with torch.no_grad():
    test_accuracy = evaluation(test_mask, num_hops, 'test')
    print(flush=True)
    if type(model) == GBPN:
        print(model.bp_conv.get_logH().exp())

with torch.no_grad():
    log_b0 = model.transform(x, edge_index)
    edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
    agg_scaling = torch.ones(x.shape[0]).to(x.device)
    info = {'log_b0': log_b0,
            'log_msg_': (-np.log(num_classes)) * torch.ones(edge_index.shape[1], num_classes).to(x.device),
            'rv': rv,
            'agg_scaling': agg_scaling}
    log_b = log_b0

    all_hops = range(20)
    log_b_list = []
    for _ in all_hops:
        log_b_list.append(log_b)
        log_b = model.bp_conv(log_b, edge_index, edge_weight, info)

    train_loss = [F.nll_loss(log_b_[train_mask], y[train_mask]) for log_b_ in log_b_list]
    train_accuracies = [accuracy_fun(log_b_[train_mask], y[train_mask]) for log_b_ in log_b_list]
    test_loss = [F.nll_loss(log_b_[test_mask], y[test_mask]) for log_b_ in log_b_list]
    test_accuracies = [accuracy_fun(log_b_[test_mask], y[test_mask]) for log_b_ in log_b_list]
    b_delta = [float((((log_b_.exp() - log_b.exp())**2).mean())**0.5) for log_b_ in log_b_list]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('hop-#')
    ax1.set_ylabel('accuracy')
    ax1.plot(all_hops, train_accuracies, color='tab:red', linestyle='solid', marker='o')
    ax1.plot(all_hops, test_accuracies, color='tab:green', linestyle='solid', marker='o')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.plot(all_hops, train_loss, color='tab:red', linestyle='dotted', marker='.')
    ax2.plot(all_hops, test_loss, color='tab:green', linestyle='dotted', marker='.')

    ax3 = ax1.twinx()
    ax3.set_ylabel('delta', color='tab:blue')
    ax3.semilogy(all_hops, b_delta, color='tab:blue', linestyle='solid', marker='.')
    ax3.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.savefig('curve.pdf')
    plt.show()

    print('finished')