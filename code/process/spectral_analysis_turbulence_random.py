# This code performs a random superposed epoch analysis of turbulent spectral magnetic field quantities.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings
import sys

# conversion factors from data to cgs units
Vmag_conv = 1.0e5
Bmag_conv = 1.0e-5
m_p = 1.67262192e-24
e_inv = np.exp(-1.0)

# correlation scale calculation method (0: 1/e; 1: AC fit)
corr_calc_method = 0
# fitting method (0: broken power law; 1: bent power law)
fitting_method = 0

# Linear function
def neg_exp(x, a):
   return np.exp(-a * x)

# Single power-law function
def power_law(x, A, a1):
   return np.log(A * x**a1)

# Bent power-law function
def bent_power_law(x, A, xb, a1, a2, d):
   return np.log(A * (x/xb)**a1 / (1.0 + (x/xb)**((a1-a2)/d))**d)

# Power spectral density via the Blackman-Tukey method: AC -> window (optional) -> FFT
def PowerSpectrumBT(data_interval, N, window, apply_window):
   data_fluc = data_interval - np.mean(data_interval)
   data_AC = np.correlate(data_fluc, data_fluc, mode='full')
   if apply_window:
      data_AC_windowed = np.multiply(data_AC, window)
   else:
      data_AC_windowed = data_AC
   data_AC_windowed_shuffled = np.concatenate((data_AC_windowed[2*N:],data_AC_windowed[:2*N]))
   data_PS = np.real(np.fft.rfft(data_AC_windowed_shuffled))
   return data_PS[1:N+1], data_AC[2*N:]

# Power spectral density via the direct FFT method
def PowerSpectrumDF(data_interval, N):
   data_fluc = data_interval - np.mean(data_interval)
   data_AC = np.correlate(data_fluc, data_fluc, mode='full') / N
   data_PS = 2 * np.abs(np.fft.rfft(data_fluc))**2 / N
   return data_PS[1:], data_AC[N-1:]

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

# Import list of events (SIs or FDs), just to know how many epochs to use in each SEA
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

# Generate list of random times
n_epochs = len(FD_list) # Number of events per superposed epoch analysis
n_seas = 500 # Number of superposed epoch analyses to perform
epoch_list = [0.0 for _ in range(n_epochs)] # List of random times

# Define parameters for turbulent analysis
n_pts = 8*1350+1 # number of data points per CIR event (each centered at SI)
half_n_pts = n_pts // 2 # half of n_pts
bin_full_width = 2700+1 # number of data points in each subinterval
bin_half_width = bin_full_width // 2 # half of bin_full_width
bin_disp = 225 # number of data point shift between each subinterval
n_intervals = n_pts // bin_disp + 1 # number of (overlapping) subintervals
bin_pad = (n_pts % bin_disp) // 2 # number of datapoints to pad on first and last subinterval centers
missing_threshold = int(0.2 * bin_full_width) # threshold of number of missing datapoints above which interval is discarded 
dt = 64.0 # Sampling time
time_lag = dt * np.arange(bin_full_width)
freq_range = np.fft.rfftfreq(bin_full_width, d=dt)[1:] # full frequency range
# Find frequency subranges
ps_acc_idx = 1 # "safe" index above which PS calculation can be trusted
acc_freq = freq_range[ps_acc_idx] # "safe" frequency above which PS calculation can be trusted
corr_freq_low = 2.0e-4 # "safe" upper bound for energy containing range (low estimate of correlation scale)
corr_freq_upp = 5.0e-4 # "safe" lower bound for inertial range (high estimate of correlation scale)
diss_freq_low = 5.0e-3 # "safe" upper bound for inertial range (low estimate of dissipation scale)
for f_idx in range(bin_full_width-1):
   if freq_range[f_idx] > corr_freq_low:
      corr_idx_low = f_idx-1
      break
for f_idx in range(corr_idx_low, bin_full_width-1):
   if freq_range[f_idx] > corr_freq_upp:
      corr_idx_upp = f_idx
      break
for f_idx in range(corr_idx_upp, bin_full_width-1):
   if freq_range[f_idx] > diss_freq_low:
      diss_idx_low = f_idx-1
      break
if fitting_method == 0:
   freq_range1 = freq_range[ps_acc_idx:corr_idx_low] # energy containing frequency range for fit
   n_freq1 = np.size(freq_range1) # size of energy containing frequency range for fit
   freq_range2 = freq_range[corr_idx_upp:diss_idx_low] # inertial frequency range for fit
   n_freq2 = np.size(freq_range2) # size of inertial frequency range for fit
else:
   freq_range3 = freq_range[ps_acc_idx:diss_idx_low]

sea_counts_list = np.zeros((n_intervals, n_seas)) # counts of samples in superposed epoch analysis
bfld_turb_eng_dens_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent magnetic field energy density
corr_length_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged correlation length
bend_length_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged bendover length
injection_range_slope_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged energy containing range slope
inertial_range_slope_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged inertial range slope
blackman_width = 4*(bin_full_width//5)+1 # blackman filter width
blackman_pad = (2*bin_full_width-1 - blackman_width)//2 # width to pad filter with zeros on either side
blackman_window = np.concatenate((np.zeros(blackman_pad),np.blackman(blackman_width),np.zeros(blackman_pad))) # blackman filter with pads
blackman_window = blackman_window / bin_full_width # scale by bin width for proper PS calculation
acceptable_responses_y = ["y", "Y", "yes", "Yes", "YES"] # acceptable affirmative responses
acceptable_responses_n = ["n", "N", "no" , "No" , "NO" ] # acceptable negative responses
acceptable_responses = acceptable_responses_y + acceptable_responses_n # acceptable responses

# Iterate over SI list to find turbulent quantities, and output results to file
epoch_idx = 0 # zero epoch index
for sea_idx in range(n_seas):
   print("Random SEA", sea_idx)
   for epoch_list_idx in range(n_epochs):
      epoch_list[epoch_list_idx] = np.random.uniform(year_start+0.025,year_end+0.975) # Generate a new list of random times
   epoch_list.sort() # sort list of random times
   epoch_idx = 0 # zero epoch index
   for epoch_list_idx in range(n_epochs):
      while year_SW[epoch_idx] < epoch_list[epoch_list_idx]:
         epoch_idx = epoch_idx + 1
      Br_CIR = SW_data[0][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Bt_CIR = SW_data[1][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Bn_CIR = SW_data[2][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Vr_CIR = SW_data[3][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Vt_CIR = SW_data[4][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Vn_CIR = SW_data[5][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Np_CIR = SW_data[6][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      B_missing_CIR = SW_data[8][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
# iterate over subintervals
      for bin_idx in range(n_intervals):
         bin_center = bin_half_width + bin_pad + bin_idx * bin_disp
         bin_left = bin_center - bin_half_width
         bin_right = bin_center + bin_half_width
# check missing data threshold
         if np.sum(B_missing_CIR[bin_left:bin_right+1] > missing_threshold):
            continue
# Extract interval quantities
         Br_interval = Br_CIR[bin_left:bin_right+1]
         Bt_interval = Bt_CIR[bin_left:bin_right+1]
         Bn_interval = Bn_CIR[bin_left:bin_right+1]
         Vr_interval = Vr_CIR[bin_left:bin_right+1]
         Vt_interval = Vt_CIR[bin_left:bin_right+1]
         Vn_interval = Vn_CIR[bin_left:bin_right+1]
         Np_interval = Np_CIR[bin_left:bin_right+1]

# total power spectrum is sum of component power spectra
         Br_PS, Br_AC = PowerSpectrumDF(Br_interval, bin_full_width)
         Bt_PS, Bt_AC = PowerSpectrumDF(Bt_interval, bin_full_width)
         Bn_PS, Bn_AC = PowerSpectrumDF(Bn_interval, bin_full_width)
         B_tot_PS = dt * (Br_PS + Bt_PS + Bn_PS) # Multiply by sampling time for proper scaling and units
         B_tot_AC = (Br_AC + Bt_AC + Bn_AC) # Divide by number of elements in interval for proper scaling
         B_var = np.var(Br_interval) + np.var(Bt_interval) + np.var(Bn_interval) # Variance of magnetic field

# find correlation time
         use_sample = False
         for lag in range(bin_full_width):
            if B_tot_AC[lag] < B_var*e_inv:
               corr_time = dt * (lag - (B_var*e_inv - B_tot_AC[lag]) / (B_tot_AC[lag-1] - B_tot_AC[lag]))
               AC_pos_lag_idx = lag
               use_sample = True
               break
         
         if use_sample:
            if corr_calc_method == 1:
               while (AC_pos_lag_idx < bin_full_width) and (B_tot_AC[AC_pos_lag_idx] > 0.0):
                  AC_pos_lag_idx = AC_pos_lag_idx+1
               guess_params = [1.0/corr_time]
               low_bound = [0.01/corr_time]
               high_bound = [100.0/corr_time]
               try_to_fit = True
               while try_to_fit:
                  try:
                     opt_params, cov = curve_fit(neg_exp, time_lag[:AC_pos_lag_idx], B_tot_AC[:AC_pos_lag_idx]/B_var,
                                                 p0=guess_params, maxfev=10000, bounds=(low_bound, high_bound))
                  except (ValueError, RuntimeError) as err:
                     print("\t\tAC fit error:", err)
                     use_sample = False
                     try_to_fit = False
                  else:
                     corr_time = 1.0/opt_params[0]
                     use_sample = True
                     try_to_fit = False

# fit bent power law function in log space
         if use_sample:
            if fitting_method == 0:
               B_tot_PS1 = B_tot_PS[ps_acc_idx:corr_idx_low]   # Power spectrum in energy contianing range
               B_tot_PS2 = B_tot_PS[corr_idx_upp:diss_idx_low] # Power spectrum in inertial range
               guess_params1 = [B_tot_PS1[0], -1.0]         # Initial guess parameters for energy containing range spectral fit
               guess_params2 = [B_tot_PS2[0]*freq_range2[0]**(5.0/3.0), -5.0/3.0] # Initial guess parameters for inertial range spectral fit
            else:
               B_tot_PS3 = B_tot_PS[ps_acc_idx:diss_idx_low]
               guess_params3 = [B_tot_PS3[0], corr_freq_low, -1.0, -5.0/3.0, 0.5] # Initial guess parameters for full spectral fit
            use_sample = False
            try_to_fit = True
            while try_to_fit:
               try:
                  if fitting_method == 0:
                     opt_params1, cov1 = curve_fit(power_law, freq_range1, np.log(B_tot_PS1), p0=guess_params1,
                                                   maxfev=10000, bounds=([0.0,-2.0],[np.inf,0.0]))
                     opt_params2, cov2 = curve_fit(power_law, freq_range2, np.log(B_tot_PS2), p0=guess_params2,
                                                   maxfev=10000, bounds=([0.0,-8.0/3.0],[np.inf,-2.0/3.0]))
                  else:
                     opt_params3, cov3 = curve_fit(bent_power_law, freq_range3, np.log(B_tot_PS3), p0=guess_params3,
                                                   maxfev=10000, bounds=([0.0,acc_freq,-2.0,-8.0/3.0,0.1],[np.inf,diss_freq_low,0.0,-2.0/3.0,1.0]))
# if curve fit fails, choose to retry fit with different initial parameters or skip this interval in overall analysis
               except (ValueError, RuntimeError) as err:
                  use_sample = False
                  try_to_fit = False
               else:
                  use_sample = True
                  try_to_fit = False

         if use_sample:
# Extract relevant parameters
            if fitting_method == 0:
               bendover_freq = (opt_params2[0]/opt_params1[0])**(1.0/(opt_params1[1]-opt_params2[1]))
               if bendover_freq < acc_freq:
                  continue
               elif bendover_freq > diss_freq_low:
                  continue
               slope1 = opt_params1[1]
               slope2 = opt_params2[1]
            else:
               bendover_freq = opt_params3[1]
               slope1 = opt_params3[2]
               slope2 = opt_params3[3]

# Add to superposed epoch average
            V_avg = np.sqrt(np.mean(Vr_interval)**2+np.mean(Vt_interval)**2+np.mean(Vn_interval)**2) # Average velocity
            sea_counts_list[bin_idx,sea_idx] = sea_counts_list[bin_idx,sea_idx] + 1.0
            bfld_turb_eng_dens_avg_list[bin_idx,sea_idx] = bfld_turb_eng_dens_avg_list[bin_idx,sea_idx] + (Bmag_conv**2) * B_var
            corr_length_avg_list[bin_idx,sea_idx] = corr_length_avg_list[bin_idx,sea_idx] + np.log(Vmag_conv * V_avg * corr_time)
            bend_length_avg_list[bin_idx,sea_idx] = bend_length_avg_list[bin_idx,sea_idx] + np.log(Vmag_conv * V_avg /  bendover_freq)
            injection_range_slope_avg_list[bin_idx,sea_idx] = injection_range_slope_avg_list[bin_idx,sea_idx] + slope1
            inertial_range_slope_avg_list[bin_idx,sea_idx] = inertial_range_slope_avg_list[bin_idx,sea_idx] + slope2

# Normalize by number of samples
   bfld_turb_eng_dens_avg_list[:,sea_idx] = np.divide(bfld_turb_eng_dens_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   corr_length_avg_list[:,sea_idx] = np.exp(np.divide(corr_length_avg_list[:,sea_idx],sea_counts_list[:,sea_idx]))
   bend_length_avg_list[:,sea_idx] = np.exp(np.divide(bend_length_avg_list[:,sea_idx],sea_counts_list[:,sea_idx]))
   injection_range_slope_avg_list[:,sea_idx] = np.divide(injection_range_slope_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   inertial_range_slope_avg_list[:,sea_idx] = np.divide(inertial_range_slope_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
# Find averages and standard deviations
bfld_turb_eng_dens_avg = np.mean(bfld_turb_eng_dens_avg_list, axis=1) # average
corr_length_avg = np.mean(corr_length_avg_list, axis=1) # average
bend_length_avg = np.mean(bend_length_avg_list, axis=1) # average
injection_range_slope_avg = np.mean(injection_range_slope_avg_list, axis=1) # average
inertial_range_slope_avg = np.mean(inertial_range_slope_avg_list, axis=1) # average
bfld_turb_eng_dens_std = np.std(bfld_turb_eng_dens_avg_list, axis=1) # standard deviation
corr_length_std = np.std(corr_length_avg_list, axis=1) # standard deviation
bend_length_std = np.std(bend_length_avg_list, axis=1) # standard deviation
injection_range_slope_std = np.std(injection_range_slope_avg_list, axis=1) # standard deviation
inertial_range_slope_std = np.std(inertial_range_slope_avg_list, axis=1) # standard deviation

# Plot
init_epoch = -4.0+(bin_pad/1350)
final_epoch = init_epoch + (n_intervals-1)*(bin_disp/1350)
epoch_time = np.linspace(init_epoch,final_epoch, num=n_intervals)
fig = plt.figure(figsize=(12, 10), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(epoch_time, corr_length_avg)
ax1.plot(epoch_time, corr_length_avg+corr_length_std)
ax1.plot(epoch_time, corr_length_avg-corr_length_std)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Spectral Correlation Length', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(epoch_time, bend_length_avg)
ax2.plot(epoch_time, bend_length_avg+bend_length_std)
ax2.plot(epoch_time, bend_length_avg-bend_length_std)
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('Spectral Bendover Length', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.axvline(0.0, color='k')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

ax3.plot(epoch_time, injection_range_slope_avg)
ax3.plot(epoch_time, injection_range_slope_avg+injection_range_slope_std)
ax3.plot(epoch_time, injection_range_slope_avg-injection_range_slope_std)
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('Injection Range Power Slope', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='k')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

ax4.plot(epoch_time, inertial_range_slope_avg)
ax4.plot(epoch_time, inertial_range_slope_avg+inertial_range_slope_std)
ax4.plot(epoch_time, inertial_range_slope_avg-inertial_range_slope_std)
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('Inertial Range Power Slope', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='k')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
if corr_calc_method == 0:
   AC_label = "AC_no_fit_"
else:
   AC_label = "AC_exp_fit_"
if fitting_method == 0:
   FM_label = "PS_broken_fit_"
else:
   FM_label = "PS_bent_fit_"
file = open("output/SEA_micro_" + AC_label + FM_label + str(year_start) + "-" + str(year_end) + "_random.lst", "w")
for bin_idx in range(n_intervals):
   file.write("{:18.6f}".format(epoch_time[bin_idx]))
   file.write("{:12.3e}{:12.3e}".format(bfld_turb_eng_dens_avg[bin_idx],bfld_turb_eng_dens_std[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}".format(corr_length_avg[bin_idx],corr_length_std[bin_idx],
                                                        bend_length_avg[bin_idx],bend_length_std[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}\n".format(injection_range_slope_avg[bin_idx],injection_range_slope_std[bin_idx],
                                                          inertial_range_slope_avg[bin_idx],inertial_range_slope_std[bin_idx]))
file.close()
print("Superposed epoch analysis results for turbulent quantities saved to disk.")
