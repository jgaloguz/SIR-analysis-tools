# This code performs a random superposed epoch analysis of the percentage change in the cosmic ray count rate.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import clean data
labels = ["c/s"] # label for each input
year_CR = [] # year (float) array
CR_data = [[] for _ in range(len(labels))] # Solar wind data inputs
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

# Import list of events (SIs or FDs), just to know how many epochs to use in each SEA
FD_list = [] # List of events
FD_good_data = 0 # Flag to check data is valid
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   items = line.split()
   FD_good_data = int(items[1])
   if FD_good_data == 2:
      FD_list.append(float(items[0]))
   line = file.readline()
file.close() 

# Generate list of random times
n_epochs = len(FD_list) # Number of events per superposed epoch analysis
n_seas = 500 # Number of superposed epoch analyses to perform
epoch_list = [0.0 for _ in range(n_epochs)] # List of random times
print("Performing", n_seas, "SEAs, each with", n_epochs, "random events.")

# Perform multiple SEA analyses, each with a different list of random epoch times
n_pts = 8*24 # number of data points per event
half_n_pts = n_pts // 2 # half of n_pts
epoch_time = np.linspace(-4.0,4.0, num=n_pts) # epoch time in days
GCR_avgpct_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged magnetic field magnitude
for sea_idx in range(n_seas):
   print("SEA", sea_idx+1)
   for epoch_list_idx in range(n_epochs):
      epoch_list[epoch_list_idx] = np.random.uniform(year_start+0.02,year_end+0.98) # Generate a new list of random times
   epoch_list.sort() # sort list of random times
   epoch_idx = 0 # zero epoch index
   for epoch_list_idx in range(n_epochs):
      while year_CR[epoch_idx] < epoch_list[epoch_list_idx]:
         epoch_idx = epoch_idx + 1
      GCR_refcts = np.mean(CR_data[0][epoch_idx-half_n_pts:epoch_idx+half_n_pts])
      GCR_avgpct_list[:,sea_idx] = GCR_avgpct_list[:,sea_idx] \
                                 + 100.0 * (CR_data[0][epoch_idx-half_n_pts:epoch_idx+half_n_pts] - GCR_refcts) / GCR_refcts
   GCR_avgpct_list[:,sea_idx] = GCR_avgpct_list[:,sea_idx] / n_epochs # average
GCR_avgpct = np.mean(GCR_avgpct_list, axis=1) # averages
GCR_stdpct = np.std(GCR_avgpct_list, axis=1) # standard deviation

# Plot
fig = plt.figure(figsize=(10, 8), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(111, projection='rectilinear')

ax1.plot(epoch_time, GCR_avgpct)
ax1.plot(epoch_time, GCR_avgpct+GCR_stdpct)
ax1.plot(epoch_time, GCR_avgpct-GCR_stdpct)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('$\\Delta$% GCR counts', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
file = open("output/SEA_cosray_" + str(year_start) + "-" + str(year_end) + "_random.lst", "w")
for pt in range(n_pts):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:12.6f}{:12.6f}\n".format(GCR_avgpct[pt], GCR_stdpct[pt]))
file.close()
print("Superposed epoch analysis results for cosmic rays saved to disk.")