# This code is meant for a quick visual inspection of cleaned cosmic ray data.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Import clean data
labels = ["c/s"] # label for each input
year_CR = [] # year (float) array
CR_data = [[] for _ in range(len(labels))] # Solar wind data inputs
print("Available date ranges:")
print("\t(1) 20070101-20101231")
print("\t(2) 20180101-20211231")
date_range = int(input("Choose date range: "))
if date_range == 1:
   year_start = 2007 # Beginning year of clean data
   year_end = 2010 # Final year of clean data
elif date_range == 2:
   year_start = 2018 # Beginning year of clean data
   year_end = 2021 # Final year of clean data
else:
   print("Invalid choice. Must choose 1 or 2.")
   exit(1)
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

# Smooth (Savitzky-Golay filter, window size = 60, polynomial order = 5)
CR_smooth = savgol_filter(CR_data[0], 24*30, 0)
CR_plot = CR_data[0]# - CR_smooth

# Plot window
while True:
   zero_epoch = float(input("Enter zero epoch for plot: "))
   plot_width = float(input("Enter plot (half) width in days: ")) / 365.0

   # Plot
   fig = plt.figure(figsize=(10, 8), layout='tight')
   ax1 = fig.add_subplot(111, projection='rectilinear')

   ax1.plot(year_CR, CR_plot)
   ax1.set_title('ACE data', fontsize=30)
   ax1.set_ylabel('c/s', fontsize=30)
   ax1.set_xlim(zero_epoch - plot_width, zero_epoch + plot_width)
   ax1.axvline(zero_epoch, color='k')
   ax1.tick_params(axis='x', labelbottom=False)
   ax1.tick_params(axis='y', labelsize=20)

   plt.show()
   # plt.savefig("output/test.png")
   plt.close(fig)

# Exit or plot another
   again = input("Enter 0 to exit: ")
   if int(again) == 0:
      break