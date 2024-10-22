# This code plots the results of the superposed epoch analysis of bulk solar wind and magnetic field quanties.

import matplotlib.pyplot as plt
import scipy.signal as spsig
import numpy as np

# Import SEA analysis results
SIR1_data = np.loadtxt("output/SEA_macro_2007-2010.lst")
not_SIR1_data = np.loadtxt("output/SEA_macro_2007-2010_random.lst")
SIR2_data = np.loadtxt("output/SEA_macro_2018-2021.lst")
not_SIR2_data = np.loadtxt("output/SEA_macro_2018-2021_random.lst")

# Smooth data
window_size = 56 # Window size for running average
poly_ord = 0 # Polynomial order: 0,1 = simple running average
epoch_time = SIR1_data[:,0]
SIR1_Bmag_avg = spsig.savgol_filter(SIR1_data[:,2], window_size, poly_ord)
SIR1_Vmag_avg = spsig.savgol_filter(SIR1_data[:,4], window_size, poly_ord)
SIR1_Np_avg = spsig.savgol_filter(SIR1_data[:,8], window_size, poly_ord)
SIR1_Tp_avg = spsig.savgol_filter(SIR1_data[:,10] / 1000, window_size, poly_ord)
not_SIR1_Bmag_avg = spsig.savgol_filter(not_SIR1_data[:,1], window_size, poly_ord)
not_SIR1_Bmag_std = spsig.savgol_filter(not_SIR1_data[:,2], window_size, poly_ord)
not_SIR1_Vmag_avg = spsig.savgol_filter(not_SIR1_data[:,3], window_size, poly_ord)
not_SIR1_Vmag_std = spsig.savgol_filter(not_SIR1_data[:,4], window_size, poly_ord)
not_SIR1_Np_avg = spsig.savgol_filter(not_SIR1_data[:,7], window_size, poly_ord)
not_SIR1_Np_std = spsig.savgol_filter(not_SIR1_data[:,8], window_size, poly_ord)
not_SIR1_Tp_avg = spsig.savgol_filter(not_SIR1_data[:,9] / 1000, window_size, poly_ord)
not_SIR1_Tp_std = spsig.savgol_filter(not_SIR1_data[:,10] / 1000, window_size, poly_ord)
SIR2_Bmag_avg = spsig.savgol_filter(SIR2_data[:,2], window_size, poly_ord)
SIR2_Vmag_avg = spsig.savgol_filter(SIR2_data[:,4], window_size, poly_ord)
SIR2_Np_avg = spsig.savgol_filter(SIR2_data[:,8], window_size, poly_ord)
SIR2_Tp_avg = spsig.savgol_filter(SIR2_data[:,10] / 1000, window_size, poly_ord)

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
# fig.suptitle('Bulk Quantities Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(epoch_time, SIR1_Bmag_avg, 'b-')
ax1.plot(epoch_time, not_SIR1_Bmag_avg, 'g--')
ax1.plot(epoch_time, not_SIR1_Bmag_avg+2.0*not_SIR1_Bmag_std, 'g:')
ax1.plot(epoch_time, not_SIR1_Bmag_avg-2.0*not_SIR1_Bmag_std, 'g:')
ax1.plot(epoch_time, SIR2_Bmag_avg, 'c-')
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('$B$ (nT)', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.set_ylim(3.0, 11.0)
ax1.axvline(0.0, color='r', linestyle='--')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(epoch_time, SIR1_Vmag_avg, 'b-')
ax2.plot(epoch_time, not_SIR1_Vmag_avg, 'g--')
ax2.plot(epoch_time, not_SIR1_Vmag_avg+2.0*not_SIR1_Vmag_std, 'g:')
ax2.plot(epoch_time, not_SIR1_Vmag_avg-2.0*not_SIR1_Vmag_std, 'g:')
ax2.plot(epoch_time, SIR2_Vmag_avg, 'c-')
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('$V$ (km/s)', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.set_ylim(320.0, 550.0)
ax2.axvline(0.0, color='r', linestyle='--')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

ax3.plot(epoch_time, SIR1_Np_avg, 'b-')
ax3.plot(epoch_time, not_SIR1_Np_avg, 'g--')
ax3.plot(epoch_time, not_SIR1_Np_avg+2.0*not_SIR1_Np_std, 'g:')
ax3.plot(epoch_time, not_SIR1_Np_avg-2.0*not_SIR1_Np_std, 'g:')
ax3.plot(epoch_time, SIR2_Np_avg, 'c-')
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('$N_p$ (cm$^3$)', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.set_ylim(2.5, 13.0)
ax3.axvline(0.0, color='r', linestyle='--')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

ax4.plot(epoch_time, SIR1_Tp_avg, 'b-')
ax4.plot(epoch_time, not_SIR1_Tp_avg, 'g--')
ax4.plot(epoch_time, not_SIR1_Tp_avg+2.0*not_SIR1_Tp_std, 'g:')
ax4.plot(epoch_time, not_SIR1_Tp_avg-2.0*not_SIR1_Tp_std, 'g:')
ax4.plot(epoch_time, SIR2_Tp_avg, 'c-')
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('$T_p$ ($\\times 10^3$ K)', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.set_ylim(30.0, 170.0)
ax4.axvline(0.0, color='r', linestyle='--')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)
