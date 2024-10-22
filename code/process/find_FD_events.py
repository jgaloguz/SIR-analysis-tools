# This code finds events of intense galactic cosmic ray depressions.

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import sys

# Whether or not to make plots
plot_FD_times = True
plot_FD_histogram = True

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
year_CR = np.array(year_CR)
for c in range(len(labels)):
   CR_data[c] = np.array(CR_data[c])

# Derived quantities
CR_cts = CR_data[0]
n_data = np.size(CR_cts)

# Find running averages of fields
init_time = year_start # Initial time (year float) to look for CIRs
final_time = year_end+1 # Final time (year float) to look for CIRs

# Find peaks
sea_width = 4 * 24 # width of SEA analysis
avg_width = 2 * 24 # width of averaging window
half_avg_width = avg_width // 2 # half of averaging width
CR_cts_avg = spsig.savgol_filter(CR_cts, avg_width, 0) # A Savitzky-Golay filter of order zero is a running average
CR_diff = np.zeros(n_data)
for i in range(half_avg_width, n_data-half_avg_width):
   CR_diff[i] = CR_cts_avg[i-half_avg_width] - CR_cts_avg[i+half_avg_width] # Difference between backward and forward running average
peaks, peaks_prop = spsig.find_peaks(CR_diff[sea_width:n_data-sea_width],
                                     height=5, distance=sea_width) # Peaks (drops) above 5 c/s separated by 4 days
FD_list = year_CR[peaks+sea_width]

if plot_FD_times:
# Quick plot to visually assess performance of algorithm
   fig = plt.figure(figsize=(18, 6), layout='tight')
   ax = fig.add_subplot(111, projection='rectilinear')

   ax.plot(year_CR, -CR_diff)
   ax.plot(FD_list, -CR_diff[peaks+sea_width], "x", markersize=10)
   ax.set_title('Candidate GCR Events {:d}-{:d}'.format(year_start, year_end), fontsize=36)
   ax.set_ylabel('$\\Delta$ count rate (c/s)', fontsize=20)
   ax.set_xlabel('Year', fontsize=20)
   ax.set_xlim(init_time,final_time)
   ax.set_ylim(-24,15)
   ax.tick_params(axis='x', labelsize=16)
   ax.tick_params(axis='y', labelsize=16)

   plt.show()
   plt.close(fig)

if plot_FD_histogram:
# Plot histogram of FDs per year
   fig = plt.figure(figsize=(12, 8), layout='tight')
   ax = fig.add_subplot(111, projection='rectilinear')

   ax.hist(FD_list, bins=final_time-init_time, range=(init_time,final_time), color='tab:blue', edgecolor='k')
   ax.set_title('CRD Events per Year {:d}-{:d}'.format(year_start, year_end), fontsize=32)
   ax.set_ylabel('Number of CRD Events', fontsize=20)
   ax.set_xlabel('Year', fontsize=20)
   ax.set_ylim(0,37)
   ax.set_xticks(np.arange(init_time,final_time+1))
   ax.tick_params(axis='x', labelsize=16)
   ax.tick_params(axis='y', labelsize=16)

   plt.show()
   plt.close(fig)

# Output list of candidate FDs
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for FD in FD_list:
   file.write("{:12.6f}{:4d}\n".format(FD,1))
file.close()
print("List of", len(FD_list),"FD epochs saved to disk.")