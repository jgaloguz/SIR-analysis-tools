# This code performs a superposed epoch analysis of the percentage change in the cosmic ray count rate for a list of events.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import clean data
labels = ["c/s"] # label for each input
year_CR = [] # year (float) array
CR_data = [[] for _ in range(len(labels))] # Cosmic ray data inputs
year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data
file = open("clean_data/CRIS_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
line = file.readline()
while line:
   data = line.split()
   year_CR.append(float(data[0]))
   for c in range(len(labels)):
      CR_data[c].append(float(data[c+1]))
   line = file.readline()
file.close()
# Convert to numpy array for convenience
for c in range(len(labels)):
   CR_data[c] = np.array(CR_data[c])

# Import list of events (SIs or FDs)
FD_list = [] # List of FD events
FD_good_data = [] # Flag to check data is valid
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   items = line.split()
   FD_list.append(float(items[0]))
   FD_good_data.append(int(items[1]))
   line = file.readline()
file.close()
n_FD = len(FD_list)

# Iterate over event list, averaging data centered at each zero epoch
n_pts = 8*24 # number of data points per event
half_n_pts = n_pts // 2 # half of n_pts
epoch_time = np.linspace(-4.0,4.0, num=n_pts) # epoch time in days

GCR_avgpct_FD1 = np.zeros(n_pts) # superposed epoch averaged GCR percentage change 1
n_FD1 = 0 # number of events 1
GCR_avgpct_FD2 = np.zeros(n_pts) # superposed epoch averaged GCR percentage change 2
n_FD2 = 0 # number of events 2
epoch_idx = 0 # zero epoch index
for FD_idx in range(n_FD):
   while year_CR[epoch_idx] < FD_list[FD_idx]:
      epoch_idx = epoch_idx + 1

   GCR_refcts = np.mean(CR_data[0][epoch_idx-half_n_pts:epoch_idx+half_n_pts])
   if FD_good_data[FD_idx] == 1:
      GCR_avgpct_FD1 = GCR_avgpct_FD1 + 100.0 * (CR_data[0][epoch_idx-half_n_pts:epoch_idx+half_n_pts] - GCR_refcts) / GCR_refcts
      n_FD1 = n_FD1 + 1
   elif FD_good_data[FD_idx] == 2:
      GCR_avgpct_FD2 = GCR_avgpct_FD2 + 100.0 * (CR_data[0][epoch_idx-half_n_pts:epoch_idx+half_n_pts] - GCR_refcts) / GCR_refcts
      n_FD2 = n_FD2 + 1
GCR_avgpct_FD1 = GCR_avgpct_FD1 / n_FD1
GCR_avgpct_FD2 = GCR_avgpct_FD2 / n_FD2

# Plot
fig = plt.figure(figsize=(10, 8), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(111, projection='rectilinear')

ax1.plot(epoch_time, GCR_avgpct_FD1)
ax1.plot(epoch_time, GCR_avgpct_FD2)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('$\\Delta$% GCR counts', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
file = open("output/SEA_cosray_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for pt in range(n_pts):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:12.6f}".format(GCR_avgpct_FD1[pt]))
   file.write("{:12.6f}\n".format(GCR_avgpct_FD2[pt]))
file.close()
print("Superposed epoch analysis results for cosmic rays saved to disk.")