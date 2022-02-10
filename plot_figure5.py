import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = [0.710, 0.728, 0.809, 0.810, 0.871, 0.872, 0.908, 0.925, 0.933, 0.958]
bars2 = [0.816, 0.809, 0.844, 0.829, 0.858, 0.865, 0.881, 0.887, 0.881, 0.867]

# Set position of bar on X axis
r0 = np.arange(len(bars1))
r1 = [x - barWidth*0.5 for x in r0]
r2 = [x + barWidth*0.5 for x in r0]

# Make the plot
plt.figure(figsize=(9,5))
plt.bar(r1, bars1, color='#33baff', width=barWidth, edgecolor='white', label='confidence')
plt.bar(r2, bars2, color='#ff7833', width=barWidth, edgecolor='white', label='accuracy')

# Add xticks on the middle of the group bars
plt.xlabel('node percentile (ordered by increasing node degree)')
plt.xlim([-0.5, len(bars1)-0.5])
plt.xticks(r0, ['10\%', '20\%', '30\%', '40\%', '50\%', '60\%', '70\%', '80\%', '90\%', '100\%'])
plt.ylabel('confidence / accuracy')
plt.ylim([0.6, 1.0])
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])

# Create legend & Show graphic
plt.legend(ncol=3, loc=2, fontsize=15)
plt.tight_layout()
plt.savefig("calibration.svg", bbox_inches='tight')
