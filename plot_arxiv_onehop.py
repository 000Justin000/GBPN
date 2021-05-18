import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14, "text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

# overall test accuracies:  
acc = np.array([66.512,  67.124,  67.368,  68.048,  68.772,  68.939])
std = np.array([ 0.306,   0.657,   0.893,   0.147,   0.402,   0.301])

# optimal deg average
deg_avg = np.array([[  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507],
                    [  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507],
                    [  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507],
                    [  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507],
                    [  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507],
                    [  1.173,  2.355,  3.619,  5.388,  7.393,  9.936, 13.103, 17.176, 23.258, 40.507]])

# optimal pll average
pll_avg = np.array([[ -1.751, -1.524, -1.368, -1.230, -1.103, -1.049, -1.017, -0.864, -0.807, -0.764],
                    [ -1.737, -1.498, -1.335, -1.195, -1.061, -1.007, -0.979, -0.832, -0.789, -0.775],
                    [ -1.730, -1.482, -1.311, -1.175, -1.055, -1.003, -0.986, -0.841, -0.812, -0.819],
                    [ -1.720, -1.462, -1.283, -1.152, -1.035, -0.998, -0.995, -0.868, -0.862, -0.913],
                    [ -1.690, -1.416, -1.246, -1.129, -1.040, -1.040, -1.078, -0.980, -1.023, -1.188],
                    [ -1.641, -1.376, -1.221, -1.177, -1.158, -1.236, -1.355, -1.287, -1.402, -1.702]])

# optimal crs average
crs_avg = np.array([[  0.505,  0.557,  0.596,  0.625,  0.660,  0.678,  0.698,  0.744,  0.772,  0.817],
                    [  0.511,  0.561,  0.600,  0.633,  0.668,  0.684,  0.707,  0.752,  0.777,  0.821],
                    [  0.512,  0.565,  0.604,  0.636,  0.670,  0.690,  0.709,  0.753,  0.777,  0.822],
                    [  0.514,  0.568,  0.610,  0.644,  0.676,  0.697,  0.714,  0.762,  0.788,  0.831],
                    [  0.520,  0.579,  0.618,  0.653,  0.687,  0.704,  0.721,  0.768,  0.794,  0.833],
                    [  0.531,  0.584,  0.625,  0.658,  0.686,  0.704,  0.720,  0.766,  0.791,  0.830]])


def plot_loglikelihood_accuracy(nn, ll, ac, ax1_ylim, ax1_yticks, ax2_ylim, ax2_yticks, fname):
    fig, ax1 = plt.subplots(figsize=(6.0, 3.5))

    ax1.set_ylabel(r'log-likelihood', color='tab:blue', fontsize=16.5)
    ax1.set_xlim([0.5, 6.5])
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    ax1.set_xticklabels([r"full", r"$100$", r"$50$", r"$20$", r"$10$", r"$5$"])
    ax1.set_ylim(ax1_ylim)
    ax1.set_yticks(ax1_yticks)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ln1 = ax1.plot(nn, ll, color='tab:blue', linestyle='dashed', label='log-likelihood')

    ax2 = ax1.twinx()
    ax2.set_xlabel('max degree during training', fontsize=16.5)
    ax2.set_ylabel(r'accuracy ($\%$)', color='tab:red', fontsize=16.5)
    ax2.set_ylim(ax2_ylim)
    ax2.set_yticks(ax2_yticks)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ln2 = ax2.plot(nn, ac, color='tab:red', linestyle='solid', marker='o', markeredgecolor='k', markersize=6, label='accuracy')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='lower right')

    fig.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.show()


nn = np.array([1, 2, 3, 4, 5, 6])
# plot_loglikelihood_accuracy(nn, np.mean(pll_avg[:,:],  axis=1), np.mean(crs_avg[:,:],  axis=1)*100, [-1.40, -1.10], [-1.40, -1.30, -1.20, -1.10], [66.00, 69.30], [66, 67, 68, 69], "ll_acc_f.svg")
# plot_loglikelihood_accuracy(nn, np.mean(pll_avg[:,:3], axis=1), np.mean(crs_avg[:,:3], axis=1)*100, [-1.56, -1.40], [-1.55, -1.50, -1.45, -1.40], [55.00, 58.25], [55, 56, 57, 58], "ll_acc_3.svg")
# plot_loglikelihood_accuracy(nn, np.mean(pll_avg[:,3:], axis=1), np.mean(crs_avg[:,3:], axis=1)*100, [-1.40, -0.80], [-1.40, -1.20, -1.00, -0.80], [71.00, 74.00], [71, 72, 73, 74], "ll_acc_7.svg")
plot_loglikelihood_accuracy(nn, np.mean(pll_avg[:,9:], axis=1), np.mean(crs_avg[:,9:], axis=1)*100, [-1.40, -0.80], [-1.40, -1.20, -1.00, -0.80], [79.00, 84.00], [71, 72, 73, 74], "ll_acc_6.svg")
