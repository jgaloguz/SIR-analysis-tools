# This code finds stream interaction regions, in a broad sense, by spotting sustained, sudden increases in solar wind speed.

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import sys

# Whether or not to make plots
plot_SI_times = True
plot_SI_histogram = True

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
year_SW = np.array(year_SW)
for c in range(len(labels)):
   SW_data[c] = np.array(SW_data[c])

# Derive quantities
Vmag = np.sqrt(SW_data[3]**2 + SW_data[4]**2 + SW_data[5]**2)
B_missing = SW_data[8]
V_missing = SW_data[9]
n_data = np.size(Vmag)

# Find running averages of fields
init_time = year_start # Initial time (year float) to look for CIRs
final_time = year_end+1 # Final time (year float) to look for CIRs

n_pts = 8*1350+1 # number of data points per CIR event (each centered at SI)
half_n_pts = n_pts // 2 # half of n_pts
avg_width = 2*1350 # width of averaging window
half_avg_width = avg_width // 2 # half of averaging width
max_missing_B = 1*1350 # maximum allowed number of data points per CIR event with missing B
max_missing_V = 2*1350 # maximum allowed number of data points per CIR event with missing V

# Find peaks
sea_width = 4*1350 # width of SEA analysis
vmag_avg = spsig.savgol_filter(Vmag, avg_width, 0) # A Savitzky-Golay filter of order zero is a running average
vmag_diff = np.zeros(n_data)
for i in range(half_avg_width, n_data-half_avg_width):
   vmag_diff[i] = vmag_avg[i+half_avg_width] - vmag_avg[i-half_avg_width] # Difference between forward and backward running average
peaks, peaks_prop = spsig.find_peaks(vmag_diff[sea_width:n_data-sea_width],
                                     height=50, distance=sea_width) # Peaks (jumps) above 50 km/s and separated by 4 days
SI_list = year_SW[peaks+sea_width]
n_SI = np.size(SI_list)

# Check data
SI_good_data = []
for SI_idx in peaks:
   if np.sum(B_missing[SI_idx-half_n_pts:SI_idx+half_n_pts+1]) < max_missing_B and \
      np.sum(V_missing[SI_idx-half_n_pts:SI_idx+half_n_pts+1]) < max_missing_V:
      SI_good_data.append(1)
   else:
      SI_good_data.append(0)

if plot_SI_times:
# Quick plot to visually assess performance of algorithm
   fig = plt.figure(figsize=(18, 6), layout='tight')
   ax = fig.add_subplot(111, projection='rectilinear')

   ax.plot(year_SW, vmag_diff)
   ax.set_title('Candidate SIR Events {:d}-{:d}'.format(year_start, year_end), fontsize=36)
   ax.set_ylabel('$\\Delta V$ (km/s)', fontsize=20)
   ax.set_xlabel('Year', fontsize=20)
   ax.set_xlim(init_time,final_time)
   ax.set_ylim(-250,350)
   ax.tick_params(axis='x', labelsize=16)
   ax.tick_params(axis='y', labelsize=16)
   for SI_idx in range(n_SI):
      if SI_good_data[SI_idx]:
         ax.axvline(SI_list[SI_idx],color='k')
      else:
         ax.axvline(SI_list[SI_idx],color='r')

   plt.show()
   plt.close(fig)

if plot_SI_histogram:
# Plot histogram of SIs per year
   fig = plt.figure(figsize=(12, 8), layout='tight')
   ax = fig.add_subplot(111, projection='rectilinear')

   SI_list_good = []
   for SI_idx in range(n_SI):
      if SI_good_data[SI_idx]:
         SI_list_good.append(SI_list[SI_idx])
   ax.hist(SI_list, bins=final_time-init_time, range=(init_time,final_time), color='tab:orange', edgecolor='k', label="discarded")
   ax.hist(SI_list_good, bins=final_time-init_time, range=(init_time,final_time), color='tab:blue', edgecolor='k',label="used")
   ax.set_title('SIR Events per Year {:d}-{:d}'.format(year_start, year_end), fontsize=32)
   ax.set_ylabel('Number of SIR Events', fontsize=20)
   ax.set_xlabel('Year', fontsize=20)
   ax.set_ylim(0,45)
   ax.set_xticks(np.arange(init_time,final_time+1))
   ax.tick_params(axis='x', labelsize=16)
   ax.tick_params(axis='y', labelsize=16)
   ax.legend(fontsize=16)

   plt.show()
   plt.close(fig)

# Output list of candidate SIs
file = open("output/SI_events_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for SI_idx in range(n_SI):
   file.write("{:12.6f}{:4d}\n".format(SI_list[SI_idx], SI_good_data[SI_idx]))
file.close()
print("List of", n_SI,"SI epochs saved to disk.")