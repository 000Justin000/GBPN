import matplotlib.pyplot as plt

nn = [1, 2, 3, 4, 5, 6]
ac = [66.512, 67.124, 67.368, 68.048, 68.772, 68.939]
ll = [-1.148, -1.121, -1.121, -1.129, -1.183, -1.356]

fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
ax1.set_xlabel('max degree during training', fontsize=16.5)
ax1.set_ylabel(r'accuracy', color='tab:blue', fontsize=16.5)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ln1 = ax1.plot(nn, ac, color='tab:blue', linestyle='solid', marker='o', markeredgecolor='k', markersize=6, label='accuracy')

ax2 = ax1.twinx()
ax2.set_ylabel(r'log-likelihood', color='tab:red', fontsize=16.5)
ax2.set_xlim([0.5, 6.5])
ax2.set_ylim([-1.4, -1.1])
ax2.set_xticks([1, 2, 3, 4, 5, 6])
ax2.set_xticklabels([r"full", r"$100$", r"$50$", r"$20$", r"$10$", r"$5$"])
ax2.set_yticks([-1.40, -1.35, -1.30, -1.25, -1.20, -1.15, -1.10])
ax2.set_yticklabels(["-1.40", "-1.35", "-1.30", "-1.25", "-1.20", "-1.15", "-1.10"])
ax2.tick_params(axis='y', labelcolor='tab:red')
ln2 = ax2.plot(nn, ll, color='tab:red', linestyle='solid', marker='o', markeredgecolor='k', markersize=6, label='log-likelihood')

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, ncol=2, loc='lower center')

fig.tight_layout()
plt.savefig('arxiv_subsampling.svg', bbox_inches='tight', pad_inches=0)
plt.show()
