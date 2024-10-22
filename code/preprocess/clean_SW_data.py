# This code takes the raw data solar wind and magnetic field data (located in "raw_data" folder) and cleans it (output goes to "clean_data" folder).
# The cleaning process consists in filling in data gaps, keeping a record of originally missing values, as well as converting time to float of year.

import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import time

# Function to fill in missing values of data by interpolating bounding valid data
def Imputate(time_array, data_array, missing_value):
   N = np.size(data_array)
# Fix first value if necessary by setting it equal to the first non-zero value
   i = 0
   while data_array[i] == missing_value:
      i = i + 1
   data_array[0] = data_array[i]
# Fix last value if necessary by setting it equal to the last non-zero value
   i = N-1
   while data_array[i] == missing_value:
      i = i - 1
   data_array[N-1] = data_array[i]
# Interpolate missing_values between first and last
   i = 1
   while i < N-1:
# If value is missing
      if data_array[i] == missing_value:
# Get last value not missing (just previous value)
         k1 = i-1
         j = i+1
         while data_array[j] == missing_value:
            j = j + 1
# Get next value not missing (guaranteed to exist)
         k2 = j
# Interpolate
         for k in range(k1+1,k2):
            data_array[k] = (data_array[k1] * (time_array[k2] - time_array[k]) + data_array[k2] * (time_array[k] - time_array[k1])) / (time_array[k2] - time_array[k1])
         i = j
      i = i + 1

# Function to record missing values of data array
def RecordMissing(data_arrays, missing_values):
   N = np.size(data_arrays,1)
   M = np.size(data_arrays,0)
   data_is_missing = np.zeros(N, dtype=np.int8)

   for i in range(N):
      for j in range(M):
         if data_arrays[j][i] == missing_values[j]:
            data_is_missing[i] = 1
            break

   return data_is_missing

# Import data with gaps
bad_value = [-9999.9,-9999.9,-9999.9,-9999.9,-9999.9,-9999.9,-9999.9,-9999.9] # Bad value flags for each input
labels = ["Np","Tp","Vr","Vt","Vn","Br","Bt","Bn"] # label for each input
year_SW = [] # year (float) array
SW_data = [[] for _ in range(len(labels))] # Solar wind data inputs
year_start = 2007 # Beginning year of raw data
year_end = 2010 # Final year of raw data
file = open("raw_data/MAG_SWEPAM_DATA_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
# Header lines
for row in range(2):
   line = file.readline()
# Data
line = file.readline()
while line:
   data = line.split()
   year_SW.append(float(data[0]))
   for c in range(len(labels)):
      SW_data[c].append(float(data[c+1]))
   line = file.readline()
file.close()

# Do a first pass to spot missing values
vel_missing = RecordMissing(SW_data[2:5], bad_value[2:5])
mag_missing = RecordMissing(SW_data[5:8], bad_value[5:8])

# Fill in bad values with linear
for c in range(len(labels)):
   Imputate(year_SW, SW_data[c], bad_value[c])

# Output clean data
file = open("clean_data/MAG_SWEPAM_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "w")
for i in range(len(year_SW)):
   file.write("{:16.8f}".format(year_SW[i]))
   file.write("{:12.3f}{:12.3f}{:12.3f}".format(SW_data[5][i],SW_data[6][i],SW_data[7][i]))
   file.write("{:12.2f}{:12.2f}{:12.2f}".format(SW_data[2][i],SW_data[3][i],SW_data[4][i]))
   file.write("{:12.3f}{:12.4e}".format(SW_data[0][i],SW_data[1][i]))
   file.write("{:4d}".format(mag_missing[i]))
   file.write("{:4d}".format(vel_missing[i]))
   file.write("\n")
file.close()