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
SIR1_Tp_avg = spsig.savgol_filter(SIR1_data[:,10] / 1e3, window_size, poly_ord)
SIR1_Pp_avg = spsig.savgol_filter(SIR1_data[:,12] * 1e12, window_size, poly_ord)
SIR1_beta_avg = spsig.savgol_filter(SIR1_data[:,14], window_size, poly_ord)
not_SIR1_Bmag_avg = spsig.savgol_filter(not_SIR1_data[:,1], window_size, poly_ord)
not_SIR1_Bmag_std = spsig.savgol_filter(not_SIR1_data[:,2], window_size, poly_ord)
not_SIR1_Vmag_avg = spsig.savgol_filter(not_SIR1_data[:,3], window_size, poly_ord)
not_SIR1_Vmag_std = spsig.savgol_filter(not_SIR1_data[:,4], window_size, poly_ord)
not_SIR1_Np_avg = spsig.savgol_filter(not_SIR1_data[:,7], window_size, poly_ord)
not_SIR1_Np_std = spsig.savgol_filter(not_SIR1_data[:,8], window_size, poly_ord)
not_SIR1_Tp_avg = spsig.savgol_filter(not_SIR1_data[:,9] / 1e3, window_size, poly_ord)
not_SIR1_Tp_std = spsig.savgol_filter(not_SIR1_data[:,10] / 1e3, window_size, poly_ord)
SIR2_Bmag_avg = spsig.savgol_filter(SIR2_data[:,2], window_size, poly_ord)
SIR2_Vmag_avg = spsig.savgol_filter(SIR2_data[:,4], window_size, poly_ord)
SIR2_Np_avg = spsig.savgol_filter(SIR2_data[:,8], window_size, poly_ord)
SIR2_Tp_avg = spsig.savgol_filter(SIR2_data[:,10] / 1e3, window_size, poly_ord)
SIR2_Pp_avg = spsig.savgol_filter(SIR2_data[:,12] * 1e12, window_size, poly_ord)
SIR2_beta_avg = spsig.savgol_filter(SIR2_data[:,14], window_size, poly_ord)

# Plot
fig = plt.figure(figsize=(8, 12), layout='tight')
# fig.suptitle('Bulk Quantities Superposed Epoch Analysis', fontsize=20)
ax1 = fig.add_subplot(321, projection='rectilinear')

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
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

ax2 = fig.add_subplot(322, projection='rectilinear')

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
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

ax3 = fig.add_subplot(323, projection='rectilinear')

ax3.plot(epoch_time, SIR1_Np_avg, 'b-')
ax3.plot(epoch_time, not_SIR1_Np_avg, 'g--')
ax3.plot(epoch_time, not_SIR1_Np_avg+2.0*not_SIR1_Np_std, 'g:')
ax3.plot(epoch_time, not_SIR1_Np_avg-2.0*not_SIR1_Np_std, 'g:')
ax3.plot(epoch_time, SIR2_Np_avg, 'c-')
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('$N_p$ (cm$^{-3}$)', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.set_ylim(2.5, 13.0)
ax3.axvline(0.0, color='r', linestyle='--')
ax3.tick_params(axis='x', labelsize=20)
ax3.tick_params(axis='y', labelsize=20)

ax4 = fig.add_subplot(324, projection='rectilinear')

ax4.plot(epoch_time, SIR1_Tp_avg, 'b-')
ax4.plot(epoch_time, not_SIR1_Tp_avg, 'g--')
ax4.plot(epoch_time, not_SIR1_Tp_avg+2.0*not_SIR1_Tp_std, 'g:')
ax4.plot(epoch_time, not_SIR1_Tp_avg-2.0*not_SIR1_Tp_std, 'g:')
ax4.plot(epoch_time, SIR2_Tp_avg, 'c-')
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('$T_p$ (kK)', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.set_ylim(30.0, 170.0)
ax4.axvline(0.0, color='r', linestyle='--')
ax4.tick_params(axis='x', labelsize=20)
ax4.tick_params(axis='y', labelsize=20)

ax5 = fig.add_subplot(325, projection='rectilinear')

ax5.plot(epoch_time, SIR1_Pp_avg, 'b-')
ax5.plot(epoch_time, SIR2_Pp_avg, 'c-')
ax5.set_xlabel('Epoch (days)', fontsize=20)
ax5.set_ylabel('$P_p$ (pPa)', fontsize=20)
ax5.set_xlim(-4.0, 4.0)
ax5.set_ylim(2.0, 22.0)
ax5.axvline(0.0, color='r', linestyle='--')
ax5.tick_params(axis='x', labelsize=20)
ax5.tick_params(axis='y', labelsize=20)

ax6 = fig.add_subplot(326, projection='rectilinear')

ax6.plot(epoch_time, SIR1_beta_avg, 'b-')
ax6.plot(epoch_time, SIR2_beta_avg, 'c-')
ax6.set_xlabel('Epoch (days)', fontsize=20)
ax6.set_ylabel('$\\beta_p$', fontsize=20)
ax6.set_xlim(-4.0, 4.0)
ax6.set_ylim(0.0, 3.5)
ax6.axvline(0.0, color='r', linestyle='--')
ax6.tick_params(axis='x', labelsize=20)
ax6.tick_params(axis='y', labelsize=20)

plt.show()
plt.close(fig)
