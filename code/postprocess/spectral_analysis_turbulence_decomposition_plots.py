# This code plots the results of the superposed epoch analysis of turbulence power and anisotropy in the solar wind and interplanetary magnetic field.

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
SIR1_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_without_SBs_2007-2010.lst")
not_SIR1_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_2007-2010_random.lst")
SIR1_turb_decomp_alt_data = np.loadtxt("output/SEA_turb_decomp_alt_without_SBs_2007-2010.lst")
SIR2_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_without_SBs_2018-2021.lst")
not_SIR2_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_2018-2021_random.lst")
SIR2_turb_decomp_alt_data = np.loadtxt("output/SEA_turb_decomp_alt_without_SBs_2018-2021.lst")

# Define data
epoch_time = SIR1_turb_decomp_data[:,0]
SIR1_bfld_eng_avg = SIR1_turb_decomp_data[:,1] / Bmag_conv**2
SIR1_bmag_eng_avg = SIR1_turb_decomp_data[:,2] / Bmag_conv**2
SIR1_alfv_eng_avg = SIR1_turb_decomp_data[:,3] / Vmag_conv**2
SIR1_vfld_eng_avg = SIR1_turb_decomp_data[:,4] / Vmag_conv**2
SIR1_P_xx_avg = SIR1_turb_decomp_data[:,5]
SIR1_P_yy_avg = SIR1_turb_decomp_data[:,6]
SIR1_P_zz_avg = SIR1_turb_decomp_data[:,7]
SIR1_P_slab_avg = SIR1_turb_decomp_data[:,8]
SIR1_P_twoD_avg = SIR1_turb_decomp_data[:,9]
not_epoch_time = not_SIR1_turb_decomp_data[:,0]
not_SIR1_bfld_eng_avg = not_SIR1_turb_decomp_data[:,1] / Bmag_conv**2
not_SIR1_bfld_eng_std = not_SIR1_turb_decomp_data[:,2] / Bmag_conv**2
not_SIR1_bmag_eng_avg = not_SIR1_turb_decomp_data[:,3] / Bmag_conv**2
not_SIR1_bmag_eng_std = not_SIR1_turb_decomp_data[:,4] / Bmag_conv**2
not_SIR1_alfv_eng_avg = not_SIR1_turb_decomp_data[:,5] / Vmag_conv**2
not_SIR1_alfv_eng_std = not_SIR1_turb_decomp_data[:,6] / Vmag_conv**2
not_SIR1_vfld_eng_avg = not_SIR1_turb_decomp_data[:,7] / Vmag_conv**2
not_SIR1_vfld_eng_std = not_SIR1_turb_decomp_data[:,8] / Vmag_conv**2
not_SIR1_P_xx_avg = not_SIR1_turb_decomp_data[:,9]
not_SIR1_P_xx_std = not_SIR1_turb_decomp_data[:,10]
not_SIR1_P_yy_avg = not_SIR1_turb_decomp_data[:,11]
not_SIR1_P_yy_std = not_SIR1_turb_decomp_data[:,12]
not_SIR1_P_zz_avg = not_SIR1_turb_decomp_data[:,13]
not_SIR1_P_zz_std = not_SIR1_turb_decomp_data[:,14]
not_SIR1_P_slab_avg = not_SIR1_turb_decomp_data[:,15]
not_SIR1_P_slab_std = not_SIR1_turb_decomp_data[:,16]
not_SIR1_P_twoD_avg = not_SIR1_turb_decomp_data[:,17]
not_SIR1_P_twoD_std = not_SIR1_turb_decomp_data[:,18]
epoch_time_alt = SIR1_turb_decomp_alt_data[:,0]
SIR1_C2_Cs = SIR1_turb_decomp_alt_data[:,2]
SIR2_bfld_eng_avg = SIR2_turb_decomp_data[:,1] / Bmag_conv**2
SIR2_bmag_eng_avg = SIR2_turb_decomp_data[:,2] / Bmag_conv**2
SIR2_alfv_eng_avg = SIR2_turb_decomp_data[:,3] / Vmag_conv**2
SIR2_vfld_eng_avg = SIR2_turb_decomp_data[:,4] / Vmag_conv**2
SIR2_P_xx_avg = SIR2_turb_decomp_data[:,5]
SIR2_P_yy_avg = SIR2_turb_decomp_data[:,6]
SIR2_P_zz_avg = SIR2_turb_decomp_data[:,7]
SIR2_P_slab_avg = SIR2_turb_decomp_data[:,8]
SIR2_P_twoD_avg = SIR2_turb_decomp_data[:,9]
SIR2_C2_Cs = SIR2_turb_decomp_alt_data[:,2]

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
# fig.suptitle('Turbulence Decomposition Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

ax1.plot(epoch_time, spsig.savgol_filter(SIR1_bfld_eng_avg, 8, 0), 'b', linestyle='-')
ax1.plot(epoch_time, spsig.savgol_filter(SIR2_bfld_eng_avg, 8, 0), 'c', linestyle='-')
ax1.plot(not_epoch_time, not_SIR1_bfld_eng_avg, 'g--')
ax1.plot(not_epoch_time, not_SIR1_bfld_eng_avg+2.0*not_SIR1_bfld_eng_std, 'g:')
ax1.plot(not_epoch_time, not_SIR1_bfld_eng_avg-2.0*not_SIR1_bfld_eng_std, 'g:')
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Magnetic Energy\nDensity (nT$^2$)', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.set_ylim(0.0, 20.0)
ax1.axvline(0.0, color='r', linestyle='--')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

ax2.plot(epoch_time, spsig.savgol_filter(SIR1_alfv_eng_avg, 8, 0), 'b', linestyle='-')
ax2.plot(not_epoch_time, not_SIR1_alfv_eng_avg, 'g--')
ax2.plot(not_epoch_time, not_SIR1_alfv_eng_avg+2.0*not_SIR1_alfv_eng_std, 'g:')
ax2.plot(not_epoch_time, not_SIR1_alfv_eng_avg-2.0*not_SIR1_alfv_eng_std, 'g:')
ax2.plot(epoch_time, spsig.savgol_filter(SIR2_alfv_eng_avg, 8, 0), 'c', linestyle='-')
ax2.plot(epoch_time, spsig.savgol_filter(SIR1_vfld_eng_avg, 8, 0), 'b', linestyle='--')
ax2.plot(epoch_time, spsig.savgol_filter(SIR2_vfld_eng_avg, 8, 0), 'c', linestyle='--')
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('Velocity Energy\nDensity (km$^2$/s$^2$)', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.set_ylim(0.0, 1200.0)
ax2.axvline(0.0, color='r', linestyle='--')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

SIR1_para_to_perp_avg = SIR1_P_zz_avg / (SIR1_P_xx_avg + SIR1_P_yy_avg)
SIR2_para_to_perp_avg = SIR2_P_zz_avg / (SIR2_P_xx_avg + SIR2_P_yy_avg)
SIR1_bmag_to_bfld_avg = 3.0 * SIR1_bmag_eng_avg / SIR1_bfld_eng_avg
SIR2_bmag_to_bfld_avg = 3.0 * SIR2_bmag_eng_avg / SIR2_bfld_eng_avg
window_size = 48
SIR1_para_to_perp_plot = spsig.savgol_filter(SIR1_para_to_perp_avg, window_size, 0)
edge_run_avg(SIR1_para_to_perp_plot, SIR1_para_to_perp_avg, window_size)
SIR2_para_to_perp_plot = spsig.savgol_filter(SIR2_para_to_perp_avg, window_size, 0)
edge_run_avg(SIR2_para_to_perp_plot, SIR2_para_to_perp_avg, window_size)
SIR1_bmag_to_bfld_plot = spsig.savgol_filter(SIR1_bmag_to_bfld_avg, window_size, 0)
edge_run_avg(SIR1_bmag_to_bfld_plot, SIR1_bmag_to_bfld_avg, window_size)
SIR2_bmag_to_bfld_plot = spsig.savgol_filter(SIR2_bmag_to_bfld_avg, window_size, 0)
edge_run_avg(SIR2_bmag_to_bfld_plot, SIR2_bmag_to_bfld_avg, window_size)

ax3.plot(epoch_time, SIR1_para_to_perp_plot, 'b', linestyle='-')
ax3.plot(epoch_time, SIR2_para_to_perp_plot, 'c', linestyle='-')
ax3.plot(epoch_time, SIR1_bmag_to_bfld_plot, 'b', linestyle='--')
ax3.plot(epoch_time, SIR2_bmag_to_bfld_plot, 'c', linestyle='--')
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('Compressibility', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.set_ylim(0.0, 0.4)
ax3.axvline(0.0, color='r', linestyle='--')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

SIR1_pct_2D_avg = 100 * SIR1_P_twoD_avg / (SIR1_P_twoD_avg + SIR1_P_slab_avg)
SIR2_pct_2D_avg = 100 * SIR2_P_twoD_avg / (SIR2_P_twoD_avg + SIR2_P_slab_avg)
window_size = 48
SIR1_pct_2D_plot = spsig.savgol_filter(SIR1_pct_2D_avg, window_size, 0)
edge_run_avg(SIR1_pct_2D_plot, SIR1_pct_2D_avg, window_size)
SIR2_pct_2D_plot = spsig.savgol_filter(SIR2_pct_2D_avg, window_size, 0)
edge_run_avg(SIR2_pct_2D_plot, SIR2_pct_2D_avg, window_size)

ax4.plot(epoch_time, SIR1_pct_2D_plot, 'b', linestyle='-')
ax4.plot(epoch_time, SIR2_pct_2D_plot, 'c', linestyle='-')
ax4.plot(epoch_time_alt, 100 * SIR1_C2_Cs / (1.0 + SIR1_C2_Cs), 'b', linestyle='--')
ax4.plot(epoch_time_alt, 100 * SIR2_C2_Cs / (1.0 + SIR2_C2_Cs), 'c', linestyle='--')
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('2D Energy (%)', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.set_ylim(0.0, 100.0)
ax4.axvline(0.0, color='r', linestyle='--')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)