# This code generates a histogram of identified sector boundaries.

import matplotlib.pyplot as plt
import scipy.signal as spsig
import numpy as np

# Import SEA analysis results
SB1_data = np.loadtxt("output/SB_epochs_2007-2010.lst")
SB2_data = np.loadtxt("output/SB_epochs_2018-2021.lst")

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
ax = fig.add_subplot(111, projection='rectilinear')

ax.hist(np.concatenate([SB1_data,SB2_data]), bins=16, range=(-4,4), color='tab:orange', edgecolor='k')
ax.hist(SB1_data, bins=16, range=(-4,4), color='tab:blue', edgecolor='k')
ax.set_ylabel('Number of SBs', fontsize=20)
ax.set_xlabel('Epoch Day', fontsize=20)
ax.set_xlim(-4,4)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)
