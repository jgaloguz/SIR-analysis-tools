# This code finds sector boundaries near the events identified in the "find_*.py" programs.

import matplotlib.pyplot as plt
import numpy as np
from TVD import TVDmm
import sys
import os

# Whether or not to make plots
plot_SB_events = True
plot_SB_histogram = False

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
avg_width = 225 # number of data points to average over
Br = np.convolve(SW_data[0], np.ones(avg_width)/avg_width, "same")
Bt = np.convolve(SW_data[1], np.ones(avg_width)/avg_width, "same")
Bn = np.convolve(SW_data[2], np.ones(avg_width)/avg_width, "same")
Bmag = np.sqrt(Br**2 + Bt**2 + Bn**2)
BV_ang_cos = np.divide(Br, Bmag)

# Import list of stream interfaces (SIs or FDs)
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
n_FD = len(FD_list)

# Remove all images in output folder
folder = 'current_sheet_crossings'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Iterate over SI list, denoising magnetic field angle data to spot sector boundaries (SB)
SB_list = [[] for _ in range(n_FD)] # List of SB for each SI
SB_epoch_list = [] # List of SBs in epoch time for plotting
n_pts = 8*1350+1 # number of data points per CIR event
half_n_pts = n_pts // 2 # half of n_pts
sector_width = 675 # Minimum number of data points of +/- angle (cosine) to constitute a magnetic sector
abs_eps = 0.175 # Absolute value threshold between reported crossings to confirm
reg_param = 75.0 # Regularization parameter
epoch_time = np.linspace(-4.0,4.0, num=n_pts) # epoch time in days
epoch_idx = 0 # zero epoch index
for FD_idx in range(n_FD):
   print("Analyzing Event", FD_idx+1)
   while year_SW[epoch_idx] < FD_list[FD_idx]:
      epoch_idx = epoch_idx + 1
   BV_ang_cos_CIR = BV_ang_cos[epoch_idx-half_n_pts-sector_width:epoch_idx+half_n_pts+sector_width+1]
   denoised_BV_ang_cos_CIR, cost = TVDmm(BV_ang_cos_CIR, reg_param, 1.0e-4, 100)

# Find SB candidates by checking where sign changes
   SB_idx_list = np.where(denoised_BV_ang_cos_CIR[1:] * denoised_BV_ang_cos_CIR[:-1] < 0.0)[0]
   SB_idx_list = SB_idx_list[(SB_idx_list >= sector_width) & (SB_idx_list < n_pts + sector_width)]
   not_SB_idx_list = []

# Filter by absolute value threshold
   for idx in SB_idx_list:
      avg_back = np.mean(denoised_BV_ang_cos_CIR[idx-sector_width:idx])
      avg_forw = np.mean(denoised_BV_ang_cos_CIR[idx:idx+sector_width])
      if avg_back * avg_forw > 0.0 or np.abs(avg_back) < abs_eps or np.abs(avg_forw) < abs_eps:
         not_SB_idx_list.append(idx)

# Assign final values
   SB_idx_list = [SB_idx - sector_width for SB_idx in SB_idx_list if SB_idx not in not_SB_idx_list]
   not_SB_idx_list = [SB_idx - sector_width for SB_idx in not_SB_idx_list]
   SB_list[FD_idx] = SB_idx_list

# Convert to epoch time
   for SB_idx in SB_idx_list:
      SB_epoch_list.append((SB_idx - half_n_pts) / 1350)

   if plot_SB_events:
# Quick plot to visually assess performance of algorithm
      fig = plt.figure(figsize=(18, 6), layout='tight')
      plt.scatter(epoch_time, BV_ang_cos_CIR[sector_width:-sector_width], s=1)
      plt.plot(epoch_time, denoised_BV_ang_cos_CIR[sector_width:-sector_width],'k', linewidth=2)
      # plt.title('Candidate Sector Crossings for Event {:d} ({:d}-{:d})'.format(FD_idx+1,year_start,year_end), fontsize=28)
      plt.ylabel('$\\cos \\theta_{B}$', fontsize=24)
      plt.xlabel('Epoch', fontsize=24)
      plt.xlim(-4.0,4.0)
      plt.ylim(-1.05,1.05)
      plt.tick_params(labelsize=24)
      plt.axhline(abs_eps, color='g', linestyle=':')
      plt.axhline(-abs_eps, color='g', linestyle=':')
      for SB_idx in SB_idx_list:
         plt.axvline(epoch_time[SB_idx], color='tab:orange', linestyle='--', linewidth=3)
      for not_SB_idx in not_SB_idx_list:
         plt.axvline(epoch_time[not_SB_idx], color='tab:red', linestyle='--', linewidth=3)
      plt.savefig("current_sheet_crossings/event_{:03d}.png".format(FD_idx+1))
      plt.close(fig)

if plot_SB_histogram:
# Plot histogram of SBs with epoch time
   fig = plt.figure(figsize=(12, 8), layout='tight')
   ax = fig.add_subplot(111, projection='rectilinear')

   ax.hist(SB_epoch_list, bins=16, range=(-4,4), color='tab:blue', edgecolor='k')
   ax.set_title('SBs per Epoch Day {:d}-{:d}'.format(year_start, year_end), fontsize=32)
   ax.set_ylabel('Number of SBs', fontsize=20)
   ax.set_xlabel('Epoch Day', fontsize=20)
   ax.tick_params(axis='x', labelsize=16)
   ax.tick_params(axis='y', labelsize=16)

   plt.show()
   plt.close(fig)

# Output list of SBs around events
file = open("output/SB_near_events_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for SB_idx_list in SB_list:
   for SB_idx in SB_idx_list:
      file.write("{:8d}".format(SB_idx))
   file.write("\n")
file.close()
file = open("output/SB_epochs_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for SB_epoch in SB_epoch_list:
   file.write("{:.6f}\n".format(SB_epoch))
file.close()
print("List of", len(SB_list),"SB around", len(FD_list), "SI saved to disk.")
