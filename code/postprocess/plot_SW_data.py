# This code is meant for a quick visual inspection of cleaned solar wind and magnetic field data.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Import clean data
labels = ["Bx","By","Bz","Vx","Vy","Vz","Np","Tp","!B","!V"] # label for each input
year_SW = [] # year (float) array
SW_data = [[] for _ in range(len(labels))] # Solar wind data inputs
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

# Plot window
while True:
   zero_epoch = float(input("Enter zero epoch for plot: "))
   plot_width = float(input("Enter plot (half) width in days: ")) / 365.0

   # Plot
   fig = plt.figure(figsize=(10, 8), layout='tight')
   ax1 = fig.add_subplot(411, projection='rectilinear')

   ax1.plot(year_SW, Bmag_smooth)
   ax1.set_title('OMNIWeb Data', fontsize=30)
   ax1.set_ylabel('Bmag', fontsize=30)
   ax1.set_ylim(0.0,15.0)
   ax1.fill_between(year_SW, 0.0, 15.0, where=B_missing==1, facecolor='red', alpha=0.5)
   ax1.set_xlim(zero_epoch - plot_width, zero_epoch + plot_width)
   ax1.axvline(zero_epoch, color='k')
   ax1.tick_params(axis='x', labelbottom=False)
   ax1.tick_params(axis='y', labelsize=20)

   ax2 = fig.add_subplot(412, projection='rectilinear', sharex=ax1)

   ax2.plot(year_SW, Vmag_smooth)
   ax2.set_ylabel('Vmag', fontsize=30)
   ax2.set_ylim(250.0,650.0)
   ax2.fill_between(year_SW, 250.0, 650.0, where=V_missing==1, facecolor='red', alpha=0.5)
   ax2.axvline(zero_epoch, color='k')
   ax2.tick_params(axis='x', labelbottom=False)
   ax2.tick_params(axis='y', labelsize=20)

   ax3 = fig.add_subplot(413, projection='rectilinear', sharex=ax1)

   ax3.plot(year_SW, Np_smooth)
   ax3.set_ylabel('Np', fontsize=30)
   ax3.set_ylim(0.0,50.0)
   ax3.fill_between(year_SW, 0.0, 50.0, where=V_missing==1, facecolor='red', alpha=0.5)
   ax3.axvline(zero_epoch, color='k')
   ax3.tick_params(axis='x', labelbottom=False)
   ax3.tick_params(axis='y', labelsize=20)

   ax4 = fig.add_subplot(414, projection='rectilinear', sharex=ax1)

   ax4.plot(year_SW, Tp_smooth)
   ax4.set_ylabel('Tp', fontsize=30)
   ax4.set_ylim(0.0,5.0e5)
   ax4.fill_between(year_SW, 0.0, 5.0e5, where=V_missing==1, facecolor='red', alpha=0.5)
   ax4.set_xlabel('Year', fontsize=30)
   ax4.axvline(zero_epoch, color='k')
   ax4.tick_params(axis='x', labelsize=15)
   ax4.tick_params(axis='y', labelsize=20)

   plt.show()
   plt.savefig("output/test.png")
   plt.close(fig)

   # Find indices bounding data
   left_idx = 0
   while year_SW[left_idx] < zero_epoch - plot_width:
      left_idx = left_idx+1
   left_idx = left_idx-1
   right_idx = left_idx+1
   while year_SW[right_idx] < zero_epoch + plot_width:
      right_idx = right_idx+1
   right_idx = right_idx+1
   # Print number of missing values
   print("Number of missing B values:", np.sum(B_missing[left_idx:right_idx]))
   print("Number of missing V values:", np.sum(V_missing[left_idx:right_idx]))

# Exit or plot another
   again = input("Enter 0 to exit: ")
   if int(again) == 0:
      break