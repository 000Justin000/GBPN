import numpy as np
import seaborn
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

def plot_coupling_potentials(H0, HH, fname):
    C0 = H0 / H0.max()
    CC = HH / HH.max()
    fig, axes = plt.subplots(1,2, figsize=(10,2.5))
    seaborn.heatmap(C0, ax=axes[0], vmin=0.0, vmax=1.0, linewidth=1.0, cmap='Blues', square=True, annot=True, fmt='5.2f', cbar=False)
    seaborn.heatmap(CC, ax=axes[1], vmin=0.0, vmax=1.0, linewidth=1.0, cmap='Blues', square=True, annot=True, fmt='5.2f', cbar=True)
    plt.savefig(fname)

H0_Ising_P = np.array([[np.exp(0.9), 1.0], [1.0, np.exp(0.9)]])
HH_Ising_P = np.array([[0.6717, 0.2545], [0.2545, 0.6725]])

H0_Ising_M = np.array([[1.0, np.exp(0.9)], [np.exp(0.9), 1.0]])
HH_Ising_M = np.array([[0.2681, 0.7361], [0.7361, 0.2707]])

H0_MRF_P = np.array([[0.9, 0.1, 0.6], [0.1, 0.9, 0.6], [0.6, 0.6, 0.2]])**(1.0/1.4)
HH_MRF_P = np.array([[0.8132, 0.1108, 0.5514], [0.1108, 0.7879, 0.5038], [0.5514, 0.5038, 0.1737]])

H0_MRF_M = np.array([[0.3, 0.1, 0.6], [0.1, 0.9, 0.1], [0.6, 0.1, 0.3]])**(1.0/1.5)
HH_MRF_M = np.array([[0.4995, 0.1159, 0.8217], [0.1159, 0.8083, 0.1228], [0.8217, 0.1228, 0.4329]])

HH_County = np.array([[0.6149, 0.3325], [0.3325, 0.5729]])
HH_Sex = np.array([[0.0998, 0.8326], [0.8326, 0.0880]])
HH_PubMed = np.array([[0.7866, 0.2095, 0.3333], [0.2095, 0.7886, 0.1717], [0.3333, 0.1717, 0.7291]])
