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

# Frequency analysis parameters
n_pts = 2700+1
dt = 64.0

# Import SEA analysis results
SIR1_macro_data = np.loadtxt("output/SEA_macro_2007-2010.lst")
SIR1_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_2007-2010.lst")

# Import data
epoch_time_macro = SIR1_macro_data[:,0]
SIR1_Vmag_avg = SIR1_macro_data[:,4]
epoch_time_micro = SIR1_micro_data[:,0]
SIR1_bfld_eng_avg = SIR1_micro_data[:,1]
SIR1_corr_length_avg = SIR1_micro_data[:,2]
SIR1_bend_length_avg = SIR1_micro_data[:,3]
SIR1_injection_range_slope_avg = SIR1_micro_data[:,4]
SIR1_inertial_range_slope_avg = SIR1_micro_data[:,5]

# Extract parameters
N_micro = np.size(epoch_time_micro)
B_var_slow = np.mean(SIR1_bfld_eng_avg[N_micro//8:3*N_micro//8])
B_var_fast = np.mean(SIR1_bfld_eng_avg[5*N_micro//8:7*N_micro//8])
lc_slow = np.mean(SIR1_corr_length_avg[N_micro//8:3*N_micro//8])
lc_fast = np.mean(SIR1_corr_length_avg[5*N_micro//8:7*N_micro//8])
lb_slow = np.mean(SIR1_bend_length_avg[N_micro//8:3*N_micro//8])
lb_fast = np.mean(SIR1_bend_length_avg[5*N_micro//8:7*N_micro//8])
inject_slow = np.mean(SIR1_injection_range_slope_avg[N_micro//8:3*N_micro//8])
inject_fast = np.mean(SIR1_injection_range_slope_avg[5*N_micro//8:7*N_micro//8])
inert_slow = np.mean(SIR1_inertial_range_slope_avg[N_micro//8:3*N_micro//8])
inert_fast = np.mean(SIR1_inertial_range_slope_avg[5*N_micro//8:7*N_micro//8])
f_min = 1.0 / dt / n_pts
f_max = 1.0 / dt / 2.0
N_macro = np.size(epoch_time_macro)
k_min_slow = f_min / np.mean(SIR1_Vmag_avg[N_macro//8:3*N_macro//8]) / Vmag_conv
k_min_fast= f_min / np.mean(SIR1_Vmag_avg[5*N_macro//8:7*N_macro//8]) / Vmag_conv
k_max_slow = f_max / np.mean(SIR1_Vmag_avg[N_macro//8:3*N_macro//8]) / Vmag_conv
k_max_fast= f_max / np.mean(SIR1_Vmag_avg[5*N_macro//8:7*N_macro//8]) / Vmag_conv

k_slow = np.array([k_min_slow, 1.0 / lb_slow, k_max_slow])
k_fast = np.array([k_min_fast, 1.0 / lb_fast, k_max_fast])

inject_slow_pls = inject_slow + 1.0
inert_slow_pls = inert_slow + 1.0
dB2_slow = (k_slow[1]**inject_slow_pls - k_slow[0]**inject_slow_pls) / inject_slow_pls \
         + (k_slow[2]**inert_slow_pls - k_slow[1]**inert_slow_pls) / inert_slow_pls
inject_fast_pls = inject_fast + 1.0
inert_fast_pls = inert_fast + 1.0
dB2_fast = (k_fast[1]**inject_fast_pls - k_fast[0]**inject_fast_pls) / inject_fast_pls \
         + (k_fast[2]**inert_fast_pls - k_fast[1]**inert_fast_pls) / inert_fast_pls

A2_slow = 3.0 * B_var_slow / dB2_slow
A2_fast = B_var_fast / dB2_fast

P_slow = A2_slow * np.array([(k_min_slow * lb_slow)**inject_slow, 1.0, (k_max_slow * lb_slow)**inert_slow])
P_fast = A2_fast * np.array([(k_min_fast * lb_fast)**inject_fast, 1.0, (k_max_fast * lb_fast)**inert_fast])

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
ax1 = fig.add_subplot(111, projection='rectilinear')

ax1.loglog(k_slow, P_slow, 'k', linestyle='-', linewidth=5)
ax1.loglog(k_fast, P_fast, 'r', linestyle='-', linewidth=5)
ax1.set_xlabel('Wavenumber (arb. unit)', fontsize=24)
ax1.set_ylabel('Magnetic Power Spectrum (arb. unit)', fontsize=24)
ax1.axvline(1.0 / lc_slow, c='k', linestyle='--', linewidth=2)
ax1.axvline(1.0 / lc_fast, c='r', linestyle='--', linewidth=2)
ax1.scatter([1.0 / lb_slow], [A2_slow], s=100.0, c='k')
ax1.scatter([1.0 / lb_fast], [A2_fast], s=100.0, c='r')
ax1.set_xlim(1e-12,1e-10)
ax1.set_ylim(1e-20,4e-17)
ax1.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
ax1.minorticks_off()

plt.show()
plt.close(fig)