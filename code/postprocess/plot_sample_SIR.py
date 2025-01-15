# This code produces a plot meant to illustrate the event selection process.

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import sys
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Function to convert a date in float of year format to date-time format
def float_year_to_datetime(year_float):
    year = np.floor(year_float).astype(int)
    remainder = year_float - year
    start_of_year = datetime(year, 1, 1)
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    seconds_from_fraction = round(remainder * days_in_year * 86400)
    return start_of_year + timedelta(seconds=seconds_from_fraction)

# Import clean data
labels_SW = ["Bx","By","Bz","Vx","Vy","Vz","Np","Tp","!B","!V"] # label for each input
year_SW = [] # year (float) array
SW_data = [[] for _ in range(len(labels_SW))] # Solar wind data inputs
labels_CR = ["c/s"] # label for each input
year_CR = [] # year (float) array
CR_data = [[] for _ in range(len(labels_CR))] # Solar wind data inputs
year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data
event_idx = int(sys.argv[2])-1 # Index of event to plot

# Solar wind
file = open("clean_data/MAG_SWEPAM_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
line = file.readline()
while line:
   data = line.split()
   year_SW.append(float(data[0]))
   for c in range(len(labels_SW)):
      if labels_SW[c][0] == "!":
         SW_data[c].append(int(data[c+1]))
      else:
         SW_data[c].append(float(data[c+1]))
   line = file.readline()
file.close()
# Convert to numpy array for convenience
for c in range(len(labels_SW)):
   SW_data[c] = np.array(SW_data[c])

# Cosmic rays
file = open("clean_data/CRIS_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
line = file.readline()
while line:
   data = line.split()
   year_CR.append(float(data[0]))
   for c in range(len(labels_CR)):
      CR_data[c].append(float(data[c+1]))
   line = file.readline()
file.close()
# Convert to numpy array for convenience
for c in range(len(labels_CR)):
   CR_data[c] = np.array(CR_data[c])

# Event
event_dates_FD = []
file = open("output/FD_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   data = line.split()
   if int(data[1]) == 2:
      event_dates_FD.append(float(data[0]))
   line = file.readline()
file.close()

event_dates_SI = []
file = open("output/SI_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
line = file.readline()
while line:
   data = line.split()
   if int(data[1]) == 2:
      event_dates_SI.append(float(data[0]))
   line = file.readline()
file.close()

# Derive quantities
Bmag = np.sqrt(SW_data[0]**2 + SW_data[1]**2 + SW_data[2]**2)
Vmag = np.sqrt(SW_data[3]**2 + SW_data[4]**2 + SW_data[5]**2)
Np = SW_data[6]
n_data = np.size(Vmag)
avg_width = 2*1350 # width of averaging window
half_avg_width = avg_width // 2 # half of averaging width
Vmag_avg = spsig.savgol_filter(Vmag, avg_width, 0) # A Savitzky-Golay filter of order zero is a running average
Vmag_diff = np.zeros(n_data)
for i in range(half_avg_width, n_data-half_avg_width):
   Vmag_diff[i] = Vmag_avg[i+half_avg_width] - Vmag_avg[i-half_avg_width] # Difference between forward and backward running average

CR_cts = CR_data[0]
n_data = np.size(CR_cts)
avg_width = 2 * 24 # width of averaging window
half_avg_width = avg_width // 2 # half of averaging width
CR_cts_avg = spsig.savgol_filter(CR_cts, avg_width, 0) # A Savitzky-Golay filter of order zero is a running average
CR_diff = np.zeros(n_data)
for i in range(half_avg_width, n_data-half_avg_width):
   CR_diff[i] = CR_cts_avg[i-half_avg_width] - CR_cts_avg[i+half_avg_width] # Difference between backward and forward running average

date_FD = event_dates_FD[event_idx]
date_SI = event_dates_SI[event_idx]
year = int(date_FD)
date_FD_dt = float_year_to_datetime(date_FD)
date_SI_dt = float_year_to_datetime(date_SI)
date_pls = float_year_to_datetime(date_FD + 3.5/365.25)
date_mns = float_year_to_datetime(date_FD - 2.5/365.25)
year_SW_dt = [float_year_to_datetime(d) for d in year_SW]
year_CR_dt = [float_year_to_datetime(d) for d in year_CR]
date_format = mdates.DateFormatter('%m/%d')

# Plot
fig = plt.figure(figsize=(16, 8), layout='tight')
# fig.suptitle('Sample Event', fontsize=20)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(year_SW_dt, Vmag, 'b-')
ax1.set_xlabel('Date (month/day of {:d})'.format(year), fontsize=20)
ax1.set_ylabel('$V$ (km/s)', fontsize=20)
ax1.set_xlim(date_mns, date_pls)
ax1.set_ylim(225.0, 775.0)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.xaxis.set_major_formatter(date_format)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(year_SW_dt, Vmag_diff, 'b-')
ax2.set_xlabel('Date (month/day of {:d})'.format(year), fontsize=20)
ax2.set_ylabel('$\\Delta V$ (km/s)', fontsize=20)
ax2.set_xlim(date_mns, date_pls)
ax2.set_ylim(-160.0, 270.0)
ax2.axvline(date_FD_dt, color='r', linestyle='--')
ax2.axvline(date_SI_dt, color='k', linestyle='--')
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.xaxis.set_major_formatter(date_format)

ax3 = fig.add_subplot(223, projection='rectilinear')

ax3.plot(year_CR_dt, CR_cts, 'b-')
ax3.set_xlabel('Date (month/day of {:d})'.format(year), fontsize=20)
ax3.set_ylabel('GCR counts (c/s)', fontsize=20)
ax3.set_xlim(date_mns, date_pls)
ax3.set_ylim(355.0, 390.0)
ax3.tick_params(axis='x', labelsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax3.xaxis.set_major_formatter(date_format)

ax4 = fig.add_subplot(224, projection='rectilinear')

ax4.plot(year_CR_dt, -CR_diff, 'b-')
ax4.set_xlabel('Date (month/day of {:d})'.format(year), fontsize=20)
ax4.set_ylabel('$\\Delta$ GCR counts (c/s)', fontsize=20)
ax4.set_xlim(date_mns, date_pls)
ax4.set_ylim(-18.0, 10.0)
ax4.axvline(date_FD_dt, color='r', linestyle='--')
ax4.axvline(date_SI_dt, color='k', linestyle='--')
ax4.tick_params(axis='x', labelsize=20)
ax4.tick_params(axis='y', labelsize=20)
ax4.xaxis.set_major_formatter(date_format)


plt.show()
plt.close(fig)
