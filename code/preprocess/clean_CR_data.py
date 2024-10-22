# This code takes the raw data cosmic ray count rate data (located in "raw_data" folder) and cleans it (output goes to "clean_data" folder).
# The cleaning process consists in filling in data gaps, keeping a record of originally missing values, as well as converting time to float of year.

import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import time

# Function to convert date (year, doy, minute) to year (float)
def toYearFraction(date):
   def sinceEpoch(date): # returns seconds since epoch
      return time.mktime(date.timetuple())
   s = sinceEpoch

   year = date.year
   startOfThisYear = dt(year=year, month=1, day=1)
   startOfNextYear = dt(year=year+1, month=1, day=1)

   yearElapsed = s(date) - s(startOfThisYear)
   yearDuration = s(startOfNextYear) - s(startOfThisYear)
   fraction = yearElapsed/yearDuration

   return date.year + fraction

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

# Import data with gaps
bad_value = [0.0] # Bad value flags for each input
labels = ["c/s"] # label for each input
year_CR = [] # year (float) array
CR_data = [[] for _ in range(len(labels))] # Solar wind data inputs
year_start = 2007 # Beginning year of raw data
year_end = 2010 # Final year of raw data
file = open("raw_data/CRIS_DATA_" + str(year_start) + "-" + str(year_end) + ".txt", "r")
# Header lines
for row in range(1):
   line = file.readline()
# Data
line = file.readline()
while line:
   data = line.split()
   date0 = dt(year=int(data[1]),month=1,day=1,hour=0,minute=0,second=0)
   date = date0 + td(days=int(data[2])-1,hours=int(data[3]),minutes=0)
   year_CR.append(toYearFraction(date))
   for c in range(len(labels)):
      CR_data[c].append(float(data[c+4]))
   line = file.readline()
file.close()

# Fill in bad values with linear
for c in range(len(labels)):
   Imputate(year_CR, CR_data[c], bad_value[c])

# Output clean data
file = open("clean_data/CRIS_DATA_clean_" + str(year_start) + "-" + str(year_end) + ".txt", "w")
for i in range(len(year_CR)):
   file.write("{:16.8f}".format(year_CR[i]))
   for c in range(len(labels)):
      file.write("{:12.2f}".format(CR_data[c][i]))
   file.write("\n")
file.close()