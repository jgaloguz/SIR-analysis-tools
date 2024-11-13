# This code performs a superposed epoch analysis of bulk solar wind and magnetic field quantities for a list of events.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Constants
kB = 1.380649e-23
mu0 = 4.0e-7 * np.pi
Np_conv = 1e6
B_conv = 1e-9

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

# Import list of events (SIs or FDs)
FD_list = [] # List of events
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

# Iterate over event list, averaging CIR data centered at each epoch
n_pts = 8*1350+1 # number of data points per event
half_n_pts = n_pts // 2 # half of n_pts
epoch_time = np.linspace(-4.0,4.0, num=n_pts) # epoch time in days

Bmag_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged magnetic field magnitude 1
Vmag_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged velocity magnitude 1
Vt_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged tangential velocity magnitude 1
Np_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged proton number density 1
Tp_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged proton temperature 1
Pp_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged proton pressure 1
beta_avg_FD1 = np.zeros(n_pts) # superposed epoch averaged proton beta 1
n_FD1 = 0 # number of events 1
Bmag_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged magnetic field magnitude 2
Vmag_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged velocity magnitude 2
Vt_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged tangential velocity magnitude 2
Np_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged proton number density 2
Tp_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged proton temperature 2
Pp_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged proton pressure 2
beta_avg_FD2 = np.zeros(n_pts) # superposed epoch averaged proton beta 2
n_FD2 = 0 # number of events 2
epoch_idx = 0 # zero epoch index
for FD_idx in range(n_FD):
   while year_SW[epoch_idx] < FD_list[FD_idx]:
      epoch_idx = epoch_idx + 1

# Extract basic quantities
   B = Bmag[epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
   Np = SW_data[6][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
   Tp = SW_data[7][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
   Pp = Np_conv * Np * kB * Tp
   beta = Pp / ((B_conv * B)**2 / (2.0 * mu0))

   if FD_good_data[FD_idx] == 1:
      Bmag_avg_FD1 = Bmag_avg_FD1 + B
      Vmag_avg_FD1 = Vmag_avg_FD1 + Vmag[epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Vt_avg_FD1 = Vt_avg_FD1 + SW_data[4][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Np_avg_FD1 = Np_avg_FD1 + Np
      Tp_avg_FD1 = Tp_avg_FD1 + Tp
      Pp_avg_FD1 = Pp_avg_FD1 + Pp
      beta_avg_FD1 = beta_avg_FD1 + beta
      n_FD1 = n_FD1 + 1
   elif FD_good_data[FD_idx] == 2:
      Bmag_avg_FD2 = Bmag_avg_FD2 + B
      Vmag_avg_FD2 = Vmag_avg_FD2 + Vmag[epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Vt_avg_FD2 = Vt_avg_FD2 + SW_data[4][epoch_idx-half_n_pts:epoch_idx+half_n_pts+1]
      Np_avg_FD2 = Np_avg_FD2 + Np
      Tp_avg_FD2 = Tp_avg_FD2 + Tp
      Pp_avg_FD2 = Pp_avg_FD2 + Pp
      beta_avg_FD2 = beta_avg_FD2 + beta
      n_FD2 = n_FD2 + 1
Bmag_avg_FD1 = Bmag_avg_FD1 / n_FD1
Vmag_avg_FD1 = Vmag_avg_FD1 / n_FD1
Vt_avg_FD1 = Vt_avg_FD1 / n_FD1
Np_avg_FD1 = Np_avg_FD1 / n_FD1
Tp_avg_FD1 = Tp_avg_FD1 / n_FD1
Pp_avg_FD1 = Pp_avg_FD1 / n_FD1
beta_avg_FD1 = beta_avg_FD1 / n_FD1
Bmag_avg_FD2 = Bmag_avg_FD2 / n_FD2
Vmag_avg_FD2 = Vmag_avg_FD2 / n_FD2
Vt_avg_FD2 = Vt_avg_FD2 / n_FD2
Np_avg_FD2 = Np_avg_FD2 / n_FD2
Tp_avg_FD2 = Tp_avg_FD2 / n_FD2
Pp_avg_FD2 = Pp_avg_FD2 / n_FD2
beta_avg_FD2 = beta_avg_FD2 / n_FD2

# Plot
fig = plt.figure(figsize=(10, 8), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(321, projection='rectilinear')

ax1.plot(epoch_time, Bmag_avg_FD1)
ax1.plot(epoch_time, Bmag_avg_FD2)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Bmag', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2a = fig.add_subplot(322, projection='rectilinear')

ax2a.plot(epoch_time, Vmag_avg_FD1)
ax2a.plot(epoch_time, Vmag_avg_FD2)
ax2a.set_xlabel('Epoch (days)', fontsize=20)
ax2a.set_ylabel('Vmag', fontsize=20)
ax2a.set_xlim(-4.0, 4.0)
ax2a.axvline(0.0, color='k')
ax2a.tick_params(axis='x', labelsize=16)
ax2a.tick_params(axis='y', labelsize=16)

ax2b = ax2a.twinx()
ax2b.plot(epoch_time, Vt_avg_FD1)
ax2b.plot(epoch_time, Vt_avg_FD2)
ax2b.set_ylabel('Vt', fontsize=20)
ax2b.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(323, projection='rectilinear')

ax3.plot(epoch_time, Np_avg_FD1)
ax3.plot(epoch_time, Np_avg_FD2)
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('Np', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='k')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(324, projection='rectilinear')

ax4.plot(epoch_time, Tp_avg_FD1)
ax4.plot(epoch_time, Tp_avg_FD2)
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('Tp', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='k')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

ax5 = fig.add_subplot(325, projection='rectilinear')

ax5.plot(epoch_time, Pp_avg_FD1)
ax5.plot(epoch_time, Pp_avg_FD2)
ax5.set_xlabel('Epoch (days)', fontsize=20)
ax5.set_ylabel('Pp', fontsize=20)
ax5.set_xlim(-4.0, 4.0)
ax5.axvline(0.0, color='k')
ax5.tick_params(axis='x', labelsize=16)
ax5.tick_params(axis='y', labelsize=16)

ax6 = fig.add_subplot(326, projection='rectilinear')

ax6.plot(epoch_time, beta_avg_FD1)
ax6.plot(epoch_time, beta_avg_FD2)
ax6.set_xlabel('Epoch (days)', fontsize=20)
ax6.set_ylabel('$\\beta$', fontsize=20)
ax6.set_xlim(-4.0, 4.0)
ax6.set_ylim(0.0, 5.0)
ax6.axvline(0.0, color='k')
ax6.tick_params(axis='x', labelsize=16)
ax6.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
file = open("output/SEA_macro_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for pt in range(n_pts):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:18.6e}{:18.6e}".format(Bmag_avg_FD1[pt], Bmag_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}".format(Vmag_avg_FD1[pt], Vmag_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}".format(Vt_avg_FD1[pt], Vt_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}".format(Np_avg_FD1[pt], Np_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}".format(Tp_avg_FD1[pt], Tp_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}".format(Pp_avg_FD1[pt], Pp_avg_FD2[pt]))
   file.write("{:18.6e}{:18.6e}\n".format(beta_avg_FD1[pt], beta_avg_FD2[pt]))
file.close()
print("Superposed epoch analysis results for bulk quantities saved to disk.")
