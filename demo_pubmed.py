import sys
import math
import subprocess
import argparse
import random
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, is_undirected, subgraph, to_networkx
from sklearn.metrics import roc_auc_score
import networkx
import umap
from utils import *
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14, "text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

def classification_accuracy(log_b, y):
    if type(log_b) == torch.Tensor:
        log_b = log_b.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    return (log_b.argmax(-1) == y).sum() / y.shape[0]


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

split = [0.3, 0.2, 0.5]
num_layers = 2
max_num_hops = 20
learning_rate = 1.0e-3
num_epoches = 300
learn_H = True
eval_C = False

# data = load_county_facebook(split=split)
# data = load_citation('Cora', split=split)
data = load_citation('PubMed', split=split)

edge_index, edge_weight, edge_rv = data.edge_index, data.edge_weight, data.edge_rv
x, y = data.x, data.y
num_nodes, num_features = x.shape
num_classes = len(torch.unique(y[y >= 0]))
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
deg = degree(edge_index[1], num_nodes)
edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32) if (edge_weight is None) else edge_weight
c_weight = None
accuracy_fun = classification_accuracy

# model = GBPN(num_features, num_classes, dim_hidden=256, num_layers=num_layers, activation=nn.LeakyReLU(), dropout_p=0.6, learn_H=learn_H)
model = GBPN(num_features, num_classes, dim_hidden=256, num_layers=num_layers, activation=nn.ReLU(), dropout_p=0.3,
                                        lossfunc_BP=5, deg_scaling=False, learn_H=learn_H)
optimizer = MultiOptimizer(torch.optim.AdamW(model.transform.parameters(), lr=learning_rate, weight_decay=2.5e-4),
                           torch.optim.AdamW(model.bp_conv.parameters(), lr=learning_rate * 10, weight_decay=2.5e-4))

graph_sampler = FullgraphSampler(num_nodes, x, y, edge_index, edge_weight, edge_rv)
if type(model) == GBPN and eval_C:
    sum_conv = SumConv()
    graphC_mask = train_mask
    graphR_mask = torch.logical_not(graphC_mask)
    graphR_edge_index, graphR_edge_weight = subgraph(graphR_mask, edge_index, edge_weight, relabel_nodes=True, num_nodes=num_nodes)
    graphR_edge_index, graphR_edge_weight, graphR_edge_rv = process_edge_index(int(graphR_mask.sum()), graphR_edge_index, graphR_edge_weight)
    graphR_sampler = type(graph_sampler)(int(graphR_mask.sum()), x[graphR_mask], y[graphR_mask], graphR_edge_index, graphR_edge_weight, graphR_edge_rv)

def train(epoch, num_hops=2):
    model.train()
    optimizer.zero_grad()
    log_b = model(x, edge_index, deg=deg, deg_ori=deg, edge_weight=edge_weight, edge_rv=edge_rv, K=num_hops)
    loss = F.nll_loss(log_b[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    accuracy = accuracy_fun(log_b[train_mask], y[train_mask])
    print('step {:5d}, train loss: {:5.3f}, train accuracy: {:5.3f}'.format(epoch, float(loss), accuracy), end='    ')
    return accuracy

def evaluation(num_hops=2):
    model.eval()
    log_b = model.inference(graph_sampler, -1, 'cpu', K=num_hops)
    train_accuracy = accuracy_fun(log_b[train_mask], y[train_mask])
    val_accuracy = accuracy_fun(log_b[val_mask], y[val_mask])
    test_accuracy = accuracy_fun(log_b[test_mask], y[test_mask])
    print('inductive accuracy: ({:5.3f}, {:5.3f}, {:5.3f})'.format(train_accuracy, val_accuracy, test_accuracy), end='    ')
    if type(model) == GBPN and eval_C:
        log_c = torch.zeros(num_nodes, num_classes)
        log_c[graphC_mask] = model.bp_conv.get_logH()[y[graphC_mask]]
        graphR_phi = sum_conv(log_c, edge_index, edge_weight)[graphR_mask]
        log_b = torch.zeros(num_nodes, num_classes)
        log_b[graphC_mask] = F.one_hot(y[graphC_mask], num_classes).float()
        log_b[graphR_mask] = model.inference(graphR_sampler, -1, 'cpu', phi=graphR_phi, K=num_hops)
        train_accuracy = accuracy_fun(log_b[train_mask], y[train_mask])
        val_accuracy = accuracy_fun(log_b[val_mask], y[val_mask])
        test_accuracy = accuracy_fun(log_b[test_mask], y[test_mask])
        print('transductive accuracy: ({:5.3f}, {:5.3f}, {:5.3f})'.format(train_accuracy, val_accuracy, test_accuracy), end='    ')
    return train_accuracy, val_accuracy, test_accuracy


run_demo = True
if run_demo:
    optimal_val_accuracy = 0.0
    optimal_test_accuracy = 0.0
    for epoch in range(1, num_epoches+1):
        num_hops = 0 if (type(model) == GBPN and epoch == 1) else max_num_hops
        train(epoch, num_hops)
        if epoch % max(int(num_epoches*0.1), 10) == 0:
            train_accuracy, val_accuracy, test_accuracy = evaluation(num_hops)
            if val_accuracy > optimal_val_accuracy:
                optimal_val_accuracy = val_accuracy
                optimal_test_accuracy = test_accuracy
                torch.save(model.state_dict(), 'model.pt')
        print(flush=True)
        if type(model) == GBPN:
            print(model.bp_conv.get_logH().exp())
    print('optimal accuracy: ({:5.3f}, {:5.3f})'.format(optimal_val_accuracy, optimal_test_accuracy))
model.load_state_dict(torch.load('model.pt'))

with torch.no_grad():
    model.eval()
    log_b0 = model.transform(x)
    edge_weight = torch.ones(edge_index.shape[1])
    info = {'log_b0': log_b0,
            'log_msg_': (-np.log(num_classes)) * torch.ones(edge_index.shape[1], num_classes),
            'edge_rv': edge_rv,
            'msg_scaling': None}
    log_b = log_b0

    all_hops = range(21)
    log_b_list = []
    for _ in all_hops:
        log_b_list.append(log_b)
        log_b = model.bp_conv(log_b, edge_index, edge_weight, info)

# plot convergence results
test_mask = torch.logical_not(train_mask)
train_loss = [F.nll_loss(log_b_[train_mask], y[train_mask]) for log_b_ in log_b_list]
train_accuracies = [accuracy_fun(log_b_[train_mask], y[train_mask]) for log_b_ in log_b_list]
test_loss = [F.nll_loss(log_b_[test_mask], y[test_mask]) for log_b_ in log_b_list]
test_accuracies = [accuracy_fun(log_b_[test_mask], y[test_mask]) for log_b_ in log_b_list]
b_delta = [float((((log_b_.exp() - log_b.exp())**2).mean())**0.5) for log_b_ in log_b_list]

fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
ax1.set_xlabel('number of BP steps ($k$)', fontsize=16.5)
ax1.set_ylabel(r'$\|p^{(k)} - p^{(20)}\|$', color='tab:blue', fontsize=16.5)
ax1.semilogy(all_hops, b_delta, color='tab:blue', linestyle='dashed', label='residual')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel(r'accuracies ($\%$)', fontsize=16.5)
ax2.set_xlim([-1, 21])
ax2.set_ylim([0.82, 0.92])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([0.82, 0.84, 0.86, 0.88, 0.90, 0.92])
ax2.set_yticklabels([82, 84, 86, 88, 90, 92])
ax2.plot(all_hops, train_accuracies, label='train', color='tab:red', linestyle='solid', marker='o', markeredgecolor='k', markersize=6)
ax2.plot(all_hops, test_accuracies, label='test', color='tab:green', linestyle='solid', marker='o', markeredgecolor='k', markersize=6)
ax2.tick_params(axis='y')
ax2.legend(loc='lower center', ncol=2)

# ax3 = ax1.twinx()
# ax3.set_ylabel('')
# ax3.set_yticks([])
# ax3.plot(all_hops, train_loss, color='tab:red', linestyle='dotted', marker='.')
# ax3.plot(all_hops, test_loss, color='tab:green', linestyle='dotted', marker='.')

fig.tight_layout()
plt.savefig('gbpn_convergence.svg', bbox_inches='tight', pad_inches=0)
plt.show()

# plot accuracy visualization
G = to_networkx(data, to_undirected=True)
N = networkx.normalized_laplacian_matrix(G)
eigvals, eigvecs = scipy.sparse.linalg.eigsh(N, k=100)
coords = umap.UMAP().fit_transform(eigvecs)
xlims = (np.min(coords[:,0])-1.5, np.max(coords[:,0])+1.5)
ylims = (np.min(coords[:,1])-1.5, np.max(coords[:,1])+1.5)

fig, ax3 = plt.subplots(figsize=(4.5,4.5))
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlim(xlims)
ax3.set_ylim(ylims)
label_colors = np.array(['tab:red', 'tab:green', 'tab:blue'])
ax3.scatter(coords[:,0], coords[:,1], c=label_colors[y.numpy()], s=5.0, alpha=0.3)

fig.tight_layout()
plt.savefig('gbpn_labels.svg', bbox_inches='tight', pad_inches=0)
plt.show()

correct_step = torch.ones(num_nodes, dtype=torch.int64)*-1
for i, log_b_ in enumerate(log_b_list):
    remaining = correct_step == -1
    correct = log_b_.argmax(dim=-1) == y
    correct_step[torch.logical_not(correct)] = -1
    correct_step[remaining & correct] = i
cmap = matplotlib.cm.get_cmap('Spectral_r', 6)

fig, ax4 = plt.subplots(figsize=(4.5,4.5))
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlim(xlims)
ax4.set_ylim(ylims)
for i in range(6):
    mask = correct_step == i
    ax4.scatter(coords[mask,0], coords[mask,1], color=cmap(i), s=5.0*0.7**i, label=r'$k={:d}$'.format(i))
    ax4.legend(loc='lower right', ncol=3, fontsize=10, markerscale=2.0, framealpha=1.0)
    fig.tight_layout()
    plt.savefig('gbpn_predictions{:d}.svg'.format(i), bbox_inches='tight', pad_inches=0)
plt.show()

print('finished')

