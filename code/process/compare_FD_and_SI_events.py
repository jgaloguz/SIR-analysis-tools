# This code finds coincident SIR and GCR depression events according to a user specified threshold.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import clean data
year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data

# Import list of events (FDs or SIs)
FD_list = [] # List of FD events
FD_good_data = [] # Flag to check data is valid
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   items = line.split()
   FD_list.append(float(items[0]))
   FD_good_data.append(1)
   line = file.readline()
file.close()
n_FD = len(FD_list)

SI_list = [] # List of SI events
SI_good_data = [] # Flag to check data is valid
file = open("output/SI_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   items = line.split()
   SI_list.append(float(items[0]))
   SI_good_data.append(int(int(items[1]) > 0))
   line = file.readline()
file.close()
n_SI = len(SI_list)

# Find matches
threshold = 1.0 / 365.25 # coincidence threshold
print("Threshold", threshold*365.25, "days")
for FD_idx in range(n_FD):
   FD = FD_list[FD_idx]
   SI_idx = (np.abs(np.array(SI_list) - FD)).argmin()
   if np.abs(FD - SI_list[SI_idx]) < threshold:
      if FD_good_data[FD_idx] and SI_good_data[SI_idx]:
         FD_good_data[FD_idx] = 2
         SI_good_data[SI_idx] = 2
      else:
         FD_good_data[FD_idx] = 0
         SI_good_data[SI_idx] = 0

# Output list of candidate FDs
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for FD_idx in range(n_FD):
   file.write("{:12.6f}{:4d}\n".format(FD_list[FD_idx], FD_good_data[FD_idx]))
file.close()
print("List of", n_FD,"FD epochs saved to disk.")
vals, cnts = np.unique(FD_good_data, return_counts=True)
print("Results", vals, cnts)

# Output list of candidate SIs
file = open("output/SI_events_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for SI_idx in range(n_SI):
   file.write("{:12.6f}{:4d}\n".format(SI_list[SI_idx], SI_good_data[SI_idx]))
file.close()
print("List of", n_SI,"SI epochs saved to disk.")
vals, cnts = np.unique(SI_good_data, return_counts=True)
print("Results", vals, cnts)
