# This code is meant for a quick visual inspection of the events identified in the "find_*.py" programs.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
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

# Smooth (Savitzky-Golay filter, window size = 60, polynomial order = 5)
Bmag_smooth = savgol_filter(Bmag, 60, 5)
Vmag_smooth = savgol_filter(Vmag, 60, 5)
Np_smooth = savgol_filter(SW_data[6], 60, 5)
Tp_smooth = savgol_filter(SW_data[7], 60, 5)
B_missing = SW_data[8]
V_missing = SW_data[9]

# Import list of stream interfaces (SIs or FDs) candidates
SI_list = [] # List of event candidates
good_data = [] # Flag to check data is valid
file = open("output/SI_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
# file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   items = line.split()
   SI_list.append(float(items[0]))
   good_data.append(int(items[1]))
   line = file.readline()
file.close()

# Plot
fig = plt.figure(figsize=(10, 8), layout='tight')
ax1 = fig.add_subplot(411, projection='rectilinear')

ax1.plot(year_SW, Bmag_smooth)
ax1.set_title('OMNIWeb Data', fontsize=30)
ax1.set_ylabel('Bmag', fontsize=30)
ax1.set_ylim(0.0,15.0)
ax1.fill_between(year_SW, 0.0, 15.0, where=B_missing==1, facecolor='red', alpha=0.5)
ax1.tick_params(axis='x', labelbottom=False)
ax1.tick_params(axis='y', labelsize=20)

ax2 = fig.add_subplot(412, projection='rectilinear', sharex=ax1)

ax2.plot(year_SW, Vmag_smooth)
ax2.set_ylabel('Vmag', fontsize=30)
ax2.set_ylim(250.0,650.0)
ax2.fill_between(year_SW, 250.0, 650.0, where=V_missing==1, facecolor='red', alpha=0.5)
ax2.tick_params(axis='x', labelbottom=False)
ax2.tick_params(axis='y', labelsize=20)

ax3 = fig.add_subplot(413, projection='rectilinear', sharex=ax1)

ax3.plot(year_SW, Np_smooth)
ax3.set_ylabel('Np', fontsize=30)
ax3.set_ylim(0.0,50.0)
ax3.fill_between(year_SW, 0.0, 50.0, where=V_missing==1, facecolor='red', alpha=0.5)
ax3.tick_params(axis='x', labelbottom=False)
ax3.tick_params(axis='y', labelsize=20)

ax4 = fig.add_subplot(414, projection='rectilinear', sharex=ax1)

ax4.plot(year_SW, Tp_smooth)
ax4.set_ylabel('Tp', fontsize=30)
ax4.fill_between(year_SW, 0.0, 5.0e5, where=V_missing==1, facecolor='red', alpha=0.5)
ax4.set_ylim(0.0,5.0e5)
ax4.set_xlabel('Year', fontsize=30)
ax4.tick_params(axis='x', labelsize=15)
ax4.tick_params(axis='y', labelsize=20)

plt.show(block=False)
plt.ion()

# Iterate over events
SI_confirmed = [] # List of confirmed events
plot_half_width = 4.0 / 365.25 # width for plotting
acceptable_responses_y = ["y", "Y", "yes", "Yes", "YES"]
acceptable_responses_n = ["n", "N", "no" , "No" , "NO" ]
acceptable_responses = acceptable_responses_y + acceptable_responses_n

for zero_epoch in SI_list:
# Preset all vertical lines marking events
   ax1.axvline(zero_epoch, color='k')
   ax2.axvline(zero_epoch, color='k')
   ax3.axvline(zero_epoch, color='k')
   ax4.axvline(zero_epoch, color='k')
left_idx = 0
for zero_epoch in SI_list:
   assessment = "maybe"
   while assessment not in acceptable_responses:
# Print zero epoch
      print("\nEpoch time =", zero_epoch)
# Update axes
      ax1.set_xlim(zero_epoch - plot_half_width, zero_epoch + plot_half_width)

# Print number of missing values
      while year_SW[left_idx] < zero_epoch - plot_half_width:
         left_idx = left_idx+1
      left_idx = left_idx-1
      right_idx = left_idx+1
      while year_SW[right_idx] < zero_epoch + plot_half_width:
         right_idx = right_idx+1
      right_idx = right_idx+1
      print("Number of missing B values:", np.sum(B_missing[left_idx:right_idx]))
      print("Number of missing V values:", np.sum(V_missing[left_idx:right_idx]))

# Decide whether to shift, accept, or reject event candidate
      assessment = input("Is this a stream interface? ")
      if assessment == "shift":
         zero_epoch = float(input("Choose new zero epoch:"))
         ax1.axvline(zero_epoch, color='k')
         ax2.axvline(zero_epoch, color='k')
         ax3.axvline(zero_epoch, color='k')
         ax4.axvline(zero_epoch, color='k')
         left_idx = left_idx - 2*24*60 # Push back left index by a couple of days in case shift was backwards
      elif assessment in acceptable_responses_y:
         SI_confirmed.append(zero_epoch)
         print("SI candidate confirmed.")
      elif assessment in acceptable_responses_n:
         print("SI candidate discarded.")
      else:
         print("Unrecognized response.")

plt.close(fig)