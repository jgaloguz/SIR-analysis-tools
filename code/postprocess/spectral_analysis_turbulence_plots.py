# This code plots the results of the superposed epoch analysis of turbulent spectral magnetic field quanties.

import matplotlib.pyplot as plt
import scipy.signal as spsig
import numpy as np

# Running average with shrinking window size around edges of data
def edge_run_avg(mod_array, org_array, window_size):
   hws = window_size//2
   for i in range(hws,0,-1):
      mod_array[i] = np.mean(org_array[:i+hws])
      mod_array[-i] = np.mean(org_array[-i-hws:])

# conversion factors from data to cgs units
Vmag_conv = 1.0e5
Bmag_conv = 1.0e-5
AU_conv = 1.496e13
mp = 1.67262192e-24

# Import SEA analysis results
SIR1_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_2007-2010.lst")
not_SIR1_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_2007-2010_random.lst")
SIR2_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_2018-2021.lst")
not_SIR2_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_2018-2021_random.lst")

# Smooth data
window_size = 4 # Window size for running average
poly_ord = 0 # Polynomial order: 0,1 = simple running average
epoch_time = SIR1_micro_data[:,0]
SIR1_corr_length_avg = spsig.savgol_filter(SIR1_micro_data[:,2] / AU_conv * 1000, window_size, poly_ord)
edge_run_avg(SIR1_corr_length_avg, SIR1_micro_data[:,2] / AU_conv * 1000, window_size)
SIR1_bend_length_avg = spsig.savgol_filter(SIR1_micro_data[:,3] / AU_conv * 1000, window_size, poly_ord)
edge_run_avg(SIR1_bend_length_avg, SIR1_micro_data[:,3] / AU_conv * 1000, window_size)
SIR1_injection_range_slope_avg = spsig.savgol_filter(SIR1_micro_data[:,4], window_size, poly_ord)
edge_run_avg(SIR1_injection_range_slope_avg, SIR1_micro_data[:,4], window_size)
SIR1_inertial_range_slope_avg = spsig.savgol_filter(SIR1_micro_data[:,5], window_size, poly_ord)
edge_run_avg(SIR1_inertial_range_slope_avg, SIR1_micro_data[:,5], window_size)
not_epoch_time = not_SIR1_micro_data[:,0]
not_SIR1_corr_length_avg = not_SIR1_micro_data[:,3] / AU_conv * 1000
not_SIR1_corr_length_std = not_SIR1_micro_data[:,4] / AU_conv * 1000
not_SIR1_bend_length_avg = not_SIR1_micro_data[:,5] / AU_conv * 1000
not_SIR1_bend_length_std = not_SIR1_micro_data[:,6] / AU_conv * 1000
not_SIR1_injection_range_slope_avg = not_SIR1_micro_data[:,7]
not_SIR1_injection_range_slope_std = not_SIR1_micro_data[:,8]
not_SIR1_inertial_range_slope_avg = not_SIR1_micro_data[:,9]
not_SIR1_inertial_range_slope_std = not_SIR1_micro_data[:,10]
SIR2_corr_length_avg = spsig.savgol_filter(SIR2_micro_data[:,2] / AU_conv * 1000, window_size, poly_ord)
edge_run_avg(SIR2_corr_length_avg, SIR2_micro_data[:,2] / AU_conv * 1000, window_size)
SIR2_bend_length_avg = spsig.savgol_filter(SIR2_micro_data[:,3] / AU_conv * 1000, window_size, poly_ord)
edge_run_avg(SIR2_bend_length_avg, SIR2_micro_data[:,3] / AU_conv * 1000, window_size)
SIR2_injection_range_slope_avg = spsig.savgol_filter(SIR2_micro_data[:,4], window_size, poly_ord)
edge_run_avg(SIR2_injection_range_slope_avg, SIR2_micro_data[:,4], window_size)
SIR2_inertial_range_slope_avg = spsig.savgol_filter(SIR2_micro_data[:,5], window_size, poly_ord)
edge_run_avg(SIR2_inertial_range_slope_avg, SIR2_micro_data[:,5], window_size)

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
# fig.suptitle('Spectral Quantities Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(epoch_time, SIR1_corr_length_avg, 'b', linestyle='-')
ax1.plot(epoch_time, SIR2_corr_length_avg, 'c', linestyle='-')
ax1.plot(not_epoch_time, not_SIR1_corr_length_avg, 'g--')
ax1.plot(not_epoch_time, not_SIR1_corr_length_avg+not_SIR1_corr_length_std, 'g:')
ax1.plot(not_epoch_time, not_SIR1_corr_length_avg-not_SIR1_corr_length_std, 'g:')
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Spectral Correlation\nScale ($\\times 10^{-3}$ au)', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='r', linestyle='--')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(epoch_time, SIR1_bend_length_avg, 'b', linestyle='-')
ax2.plot(epoch_time, SIR2_bend_length_avg, 'c', linestyle='-')
ax2.plot(not_epoch_time, not_SIR1_bend_length_avg, 'g--')
ax2.plot(not_epoch_time, not_SIR1_bend_length_avg+not_SIR1_bend_length_std, 'g:')
ax2.plot(not_epoch_time, not_SIR1_bend_length_avg-not_SIR1_bend_length_std, 'g:')
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('Spectral Bendover\nLength ($\\times 10^{-3}$ au)', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.axvline(0.0, color='r', linestyle='--')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

ax3.plot(epoch_time, SIR1_injection_range_slope_avg, 'b', linestyle='-')
ax3.plot(epoch_time, SIR2_injection_range_slope_avg, 'c', linestyle='-')
ax3.plot(not_epoch_time, not_SIR1_injection_range_slope_avg, 'g--')
ax3.plot(not_epoch_time, not_SIR1_injection_range_slope_avg+not_SIR1_injection_range_slope_std, 'g:')
ax3.plot(not_epoch_time, not_SIR1_injection_range_slope_avg-not_SIR1_injection_range_slope_std, 'g:')
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('Energy Range\nPower Slope', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='r', linestyle='--')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

ax4.plot(epoch_time, SIR1_inertial_range_slope_avg, 'b', linestyle='-')
ax4.plot(epoch_time, SIR2_inertial_range_slope_avg, 'c', linestyle='-')
ax4.plot(not_epoch_time, not_SIR1_inertial_range_slope_avg, 'g--')
ax4.plot(not_epoch_time, not_SIR1_inertial_range_slope_avg+not_SIR1_inertial_range_slope_std, 'g:')
ax4.plot(not_epoch_time, not_SIR1_inertial_range_slope_avg-not_SIR1_inertial_range_slope_std, 'g:')
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('Inertial Range\nPower Slope', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='r', linestyle='--')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)