import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pickle
import random

import scipy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import torch
from torch.nn.functional import one_hot

import umap

plt.rcParams.update({"font.size": 14, "text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

random.seed(0)
np.random.seed(0)

plot_data = pickle.load(open("plot_data.pkl", "rb"))
G = plot_data["graph"]
X = plot_data["X"]
y = plot_data["y"]
num_classes = len(np.unique(y[y >= 0]))
correct_step = plot_data["correct_step"]
num_nodes, num_features = X.shape
print(X.shape)

A = networkx.adjacency_matrix(G)
#print(sum(A))
d = np.squeeze(sum(A).toarray())
tau = sum(d) / len(d)
print(tau)
d_inv = 1.0 / np.sqrt(d + tau)
def regularized_norm_adjacency(v0):
    v1 = d_inv * v0
    s = sum(v1)
    v2 = A @ v1
    v3 = v2 + (tau / num_nodes) * s
    v4 = d_inv * v3
    return v4 + 1
M = LinearOperator((num_nodes, num_nodes), matvec=regularized_norm_adjacency)

# Spectral embedding
#N = networkx.normalized_laplacian_matrix(G)
eigvals, eigvecs = eigsh(M, k=25, which='LM')

# Feature embedding
X_pca = PCA(n_components=25).fit_transform(StandardScaler().fit_transform(X))

# one-hot class label
y_oh = StandardScaler().fit_transform(one_hot(y, num_classes=num_classes).numpy())

#Z = np.hstack((eigvecs, X_pca))
#Z = np.hstack((eigvecs, y_oh))
#Z = eigvecs
Z = np.hstack((eigvecs, X_pca, y_oh))

coords = umap.UMAP(n_neighbors=10).fit_transform(Z)
#coords = TSNE(n_components=2).fit_transform(Z)
xlims = (np.min(coords[:,0])-1.5, np.max(coords[:,0])+1.5)
ylims = (np.min(coords[:,1])-1.5, np.max(coords[:,1])+1.5)

cmap = matplotlib.cm.get_cmap('Spectral_r', 6)
fig, ax4 = plt.subplots(figsize=(4.5,4.5))
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlim(xlims)
ax4.set_ylim(ylims)
for i in range(6):
    mask = correct_step == i
    ax4.scatter(coords[mask,0], coords[mask,1], color=cmap(i), s=5.0*0.7**i, label=r'$t={:d}$'.format(i))
    ax4.legend(loc='lower right', ncol=3, fontsize=10, markerscale=2.0, framealpha=1.0)
    fig.tight_layout()
    plt.savefig('gbpn_predictions{:d}.svg'.format(i), bbox_inches='tight', pad_inches=0)

plt.savefig("gbpn_predictions.pdf", bbox_inches='tight', pad_inches=0)
plt.show()


print('finished')
