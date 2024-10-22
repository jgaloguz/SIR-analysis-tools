# This code performs a random superposed epoch analysis of bulk solar wind and magnetic field quantities.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import clean data
labels = ["Br","Bt","Bn","Vr","Vt","Vn","Np","Tp","!B","!V"] # label for each input
year_SW = [] # year (float) array
SW_data = [[] for _ in range(len(labels))] # Solar wind data inputs
year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data
file = open("clean_data/MAG_SWEPAM_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
line = file.readline()
while line:
   data = line.split()
   year_SW.append(float(data[0]))
   for c in range(len(labels)):
      if labels[c][0] == "!":
         SW_data[c].append(int(data[c+1]))
      else:
         SW_data[c].append(float(data[c+1]))
   line = file.readline()
file.close()
# Convert to numpy array for convenience
for c in range(len(labels)):
   SW_data[c] = np.array(SW_data[c])

# Derive quantities
Bmag = np.sqrt(SW_data[0]**2 + SW_data[1]**2 + SW_data[2]**2)
Vmag = np.sqrt(SW_data[3]**2 + SW_data[4]**2 + SW_data[5]**2)

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
n_pts = 8*1350+1 # number of data points per event
half_n_pts = n_pts // 2 # half of n_pts
epoch_time = np.linspace(-4.0,4.0, num=n_pts) # epoch time in days
Bmag_avg_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged magnetic field magnitude
Vmag_avg_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged velocity magnitude
Vt_avg_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged tangential velocity magnitude
Np_avg_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged proton number density
Tp_avg_list = np.zeros((n_pts, n_seas)) # superposed epoch averaged proton temperature
for sea_idx in range(n_seas):
   print("SEA", sea_idx+1)
   for epoch_list_idx in range(n_epochs):
      epoch_list[epoch_list_idx] = np.random.uniform(year_start+0.02,year_end+0.98) # Generate a new list of random times
   epoch_list.sort() # sort list of random times
   epoch_idx = 0 # zero epoch index
   for epoch_list_idx in range(n_epochs):
      while year_SW[epoch_idx] < epoch_list[epoch_list_idx]:
         epoch_idx = epoch_idx + 1
      Bmag_avg_list[:,sea_idx] = Bmag_avg_list[:,sea_idx] + Bmag[epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Vmag_avg_list[:,sea_idx] = Vmag_avg_list[:,sea_idx] + Vmag[epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Vt_avg_list[:,sea_idx] = Vt_avg_list[:,sea_idx] + SW_data[4][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Np_avg_list[:,sea_idx] = Np_avg_list[:,sea_idx] + SW_data[6][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Tp_avg_list[:,sea_idx] = Tp_avg_list[:,sea_idx] + SW_data[7][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
   Bmag_avg_list[:,sea_idx] = Bmag_avg_list[:,sea_idx] / n_epochs # average
   Vmag_avg_list[:,sea_idx] = Vmag_avg_list[:,sea_idx] / n_epochs # average
   Vt_avg_list[:,sea_idx] = Vt_avg_list[:,sea_idx] / n_epochs # average
   Np_avg_list[:,sea_idx] = Np_avg_list[:,sea_idx] / n_epochs # average
   Tp_avg_list[:,sea_idx] = Tp_avg_list[:,sea_idx] / n_epochs # average
Bmag_avg = np.mean(Bmag_avg_list, axis=1) # averages
Vmag_avg = np.mean(Vmag_avg_list, axis=1) # averages
Vt_avg = np.mean(Vt_avg_list, axis=1) # averages
Np_avg = np.mean(Np_avg_list, axis=1) # averages
Tp_avg = np.mean(Tp_avg_list, axis=1) # averages
Bmag_std = np.std(Bmag_avg_list, axis=1) # standard deviation
Vmag_std = np.std(Vmag_avg_list, axis=1) # standard deviation
Vt_std = np.std(Vt_avg_list, axis=1) # standard deviation
Np_std = np.std(Np_avg_list, axis=1) # standard deviation
Tp_std = np.std(Tp_avg_list, axis=1) # standard deviation

# Plot
fig = plt.figure(figsize=(10, 8), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(epoch_time, Bmag_avg)
ax1.plot(epoch_time, Bmag_avg+Bmag_std)
ax1.plot(epoch_time, Bmag_avg-Bmag_std)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Bmag', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(epoch_time, Vmag_avg)
ax2.plot(epoch_time, Vmag_avg+Vmag_std)
ax2.plot(epoch_time, Vmag_avg-Vmag_std)
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('Vmag', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.axvline(0.0, color='k')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

ax3.plot(epoch_time, Np_avg)
ax3.plot(epoch_time, Np_avg+Np_std)
ax3.plot(epoch_time, Np_avg-Np_std)
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('Np', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='k')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

ax4.plot(epoch_time, Tp_avg)
ax4.plot(epoch_time, Tp_avg+Tp_std)
ax4.plot(epoch_time, Tp_avg-Tp_std)
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('Tp', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='k')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
file = open("output/SEA_macro_" + str(year_start) + "-" + str(year_end) + "_random.lst", "w")
for pt in range(n_pts):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:12.2f}{:12.2f}".format(Bmag_avg[pt], Bmag_std[pt]))
   file.write("{:12.2f}{:12.2f}".format(Vmag_avg[pt], Vmag_std[pt]))
   file.write("{:12.2f}{:12.2f}".format(Vt_avg[pt], Vt_std[pt]))
   file.write("{:12.2f}{:12.2f}".format(Np_avg[pt], Np_std[pt]))
   file.write("{:12.2f}{:12.2f}\n".format(Tp_avg[pt], Tp_std[pt]))
file.close()
print("Superposed epoch analysis results for bulk quantities saved to disk.")