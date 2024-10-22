# This code plots the results of the superposed epoch analysis of cosmic ray counts and transport coefficients.

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

# Running average with shrinking window size around edges of data
def edge_run_avg(mod_array, org_array, window_size):
   hws = window_size//2
   for i in range(hws,0,-1):
      mod_array[i] = np.mean(org_array[:i+hws])
      mod_array[-i] = np.mean(org_array[-i-hws:])

# conversion factors from data to cgs units
k0_para = 1.0e22
k0_perp = 1.0e20

# Import SEA analysis results
GCR1_data = np.loadtxt("output/SEA_cosray_2007-2010.lst")
not_GCR1_data = np.loadtxt("output/SEA_cosray_2007-2010_random.lst")
GCR2_data = np.loadtxt("output/SEA_cosray_2018-2021.lst")
not_GCR2_data = np.loadtxt("output/SEA_cosray_2018-2021_random.lst")

diff1_diff_coeffs_data = np.loadtxt("output/SEA_diff_coeffs_2007-2010.lst")
diff2_diff_coeffs_data = np.loadtxt("output/SEA_diff_coeffs_2018-2021.lst")

GCR_epoch_time = GCR1_data[:,0]
GCR1_avgpct = GCR1_data[:,2]
not_GCR1_avgpct = not_GCR1_data[:,1]
not_GCR1_stdpct = not_GCR1_data[:,2]
GCR2_avgpct = GCR2_data[:,2]

diff_epoch_time = diff1_diff_coeffs_data[:,0]
diff1_kappa_para_emp = diff1_diff_coeffs_data[:,1]
diff1_kappa_para_QLT = diff1_diff_coeffs_data[:,2]
diff1_kappa_para_QLT_alt = diff1_diff_coeffs_data[:,3]
diff1_kappa_perp_LZP = diff1_diff_coeffs_data[:,4]
diff1_kappa_perp_LZP_alt = diff1_diff_coeffs_data[:,5]
diff1_kappa_perp_NLGC = diff1_diff_coeffs_data[:,6]
diff1_kappa_perp_NLGC_alt = diff1_diff_coeffs_data[:,7]
diff1_kappa_perp_FLRW = diff1_diff_coeffs_data[:,8]
diff1_kappa_perp_FLRW_alt = diff1_diff_coeffs_data[:,9]
diff2_kappa_para_emp = diff2_diff_coeffs_data[:,1]
diff2_kappa_para_QLT = diff2_diff_coeffs_data[:,2]
diff2_kappa_para_QLT_alt = diff2_diff_coeffs_data[:,3]
diff2_kappa_perp_LZP = diff2_diff_coeffs_data[:,4]
diff2_kappa_perp_LZP_alt = diff2_diff_coeffs_data[:,5]
diff2_kappa_perp_NLGC = diff2_diff_coeffs_data[:,6]
diff2_kappa_perp_NLGC_alt = diff2_diff_coeffs_data[:,7]
diff2_kappa_perp_FLRW = diff2_diff_coeffs_data[:,8]
diff2_kappa_perp_FLRW_alt = diff2_diff_coeffs_data[:,9]

# Plot
fig = plt.figure(figsize=(12, 6), layout='tight')
fig.suptitle('Diffusion Coefficients Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(231, projection='rectilinear')

ax1.plot(GCR_epoch_time, GCR1_avgpct, 'b-')
ax1.plot(GCR_epoch_time, not_GCR1_avgpct, 'g--')
ax1.plot(GCR_epoch_time, not_GCR1_avgpct+2.0*not_GCR1_stdpct, 'g:')
ax1.plot(GCR_epoch_time, not_GCR1_avgpct-2.0*not_GCR1_stdpct, 'g:')
ax1.plot(GCR_epoch_time, GCR2_avgpct, 'c-')
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('%$\\Delta$ >120 MeV H$^+$ (c/s)', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.set_ylim(-1.7, 1.7)
ax1.axvline(0.0, color='r', linestyle='--')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(232, projection='rectilinear')

diff1_kappa_para_emp_plot = spsig.savgol_filter(diff1_kappa_para_emp, 8, 0)
edge_run_avg(diff1_kappa_para_emp_plot, diff1_kappa_para_emp, 8)
diff2_kappa_para_emp_plot = spsig.savgol_filter(diff2_kappa_para_emp, 8, 0)
edge_run_avg(diff2_kappa_para_emp_plot, diff2_kappa_para_emp, 8)

ax2.plot(diff_epoch_time, diff1_kappa_para_emp_plot / k0_para, 'b', linestyle='-')
ax2.plot(diff_epoch_time, diff2_kappa_para_emp_plot / k0_para, 'c', linestyle='-')
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('$\\kappa_\\parallel$ ($\\times 10^{22}$ cm$^2$/s)', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.set_ylim(2.0, 7.0)
ax2.axvline(0.0, color='r', linestyle='--')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.text(3.0, 6.5, "EMP", color='k', fontsize=20)

ax3 = fig.add_subplot(233, projection='rectilinear')

diff1_kappa_para_QLT_plot = spsig.savgol_filter(diff1_kappa_para_QLT, 8, 0)
edge_run_avg(diff1_kappa_para_QLT_plot, diff1_kappa_para_QLT, 8)
diff2_kappa_para_QLT_plot = spsig.savgol_filter(diff2_kappa_para_QLT, 8, 0)
edge_run_avg(diff2_kappa_para_QLT_plot, diff2_kappa_para_QLT, 8)

ax3.plot(diff_epoch_time, diff1_kappa_para_QLT_plot / k0_para, 'b', linestyle='-')
ax3.plot(diff_epoch_time, diff2_kappa_para_QLT_plot / k0_para, 'c', linestyle='-')
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('$\\kappa_\\parallel$ ($\\times 10^{22}$ cm$^2$/s)', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.set_ylim(2.0, 7.0)
ax3.axvline(0.0, color='r', linestyle='--')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)
ax3.text(3.0, 6.5, "QLT", color='k', fontsize=20)

ax4 = fig.add_subplot(234, projection='rectilinear')

diff1_kappa_perp_FLRW_plot = spsig.savgol_filter(diff1_kappa_perp_FLRW, 8, 0)
edge_run_avg(diff1_kappa_perp_FLRW_plot, diff1_kappa_perp_FLRW, 8)
diff2_kappa_perp_FLRW_plot = spsig.savgol_filter(diff2_kappa_perp_FLRW, 8, 0)
edge_run_avg(diff2_kappa_perp_FLRW_plot, diff2_kappa_perp_FLRW, 8)

ax4.plot(diff_epoch_time, diff1_kappa_perp_FLRW_plot / k0_perp, 'b', linestyle='-')
ax4.plot(diff_epoch_time, diff2_kappa_perp_FLRW_plot / k0_perp, 'c', linestyle='-')
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('$\\kappa_\\perp$ ($\\times 10^{20}$ cm$^2$/s)', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.set_ylim(2.0, 5.0)
ax4.axvline(0.0, color='r', linestyle='--')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)
ax4.text(2.7, 4.7, "FLRW", color='k', fontsize=20)

ax5 = fig.add_subplot(235, projection='rectilinear')

diff1_kappa_perp_LZP_plot = spsig.savgol_filter(diff1_kappa_perp_LZP, 8, 0)
edge_run_avg(diff1_kappa_perp_LZP_plot, diff1_kappa_perp_LZP, 8)
diff2_kappa_perp_LZP_plot = spsig.savgol_filter(diff2_kappa_perp_LZP, 8, 0)
edge_run_avg(diff2_kappa_perp_LZP_plot, diff2_kappa_perp_LZP, 8)

ax5.plot(diff_epoch_time, diff1_kappa_perp_LZP_plot / k0_perp, 'b', linestyle='-')
ax5.plot(diff_epoch_time, diff2_kappa_perp_LZP_plot / k0_perp, 'c', linestyle='-')
ax5.set_xlabel('Epoch (days)', fontsize=20)
ax5.set_ylabel('$\\kappa_\\perp$ ($\\times 10^{20}$ cm$^2$/s)', fontsize=20)
ax5.set_xlim(-4.0, 4.0)
ax5.set_ylim(2.0, 5.0)
ax5.axvline(0.0, color='r', linestyle='--')
ax5.tick_params(axis='x', labelsize=16)
ax5.tick_params(axis='y', labelsize=16)
ax5.text(3.0, 4.7, "LZP", color='k', fontsize=20)

ax6 = fig.add_subplot(236, projection='rectilinear')

diff1_kappa_perp_NLGC_plot = spsig.savgol_filter(diff1_kappa_perp_NLGC, 8, 0)
edge_run_avg(diff1_kappa_perp_NLGC_plot, diff1_kappa_perp_NLGC, 8)
diff2_kappa_perp_NLGC_plot = spsig.savgol_filter(diff2_kappa_perp_NLGC, 8, 0)
edge_run_avg(diff2_kappa_perp_NLGC_plot, diff2_kappa_perp_NLGC, 8)

ax6.plot(diff_epoch_time, diff1_kappa_perp_NLGC_plot / k0_perp, 'b', linestyle='-')
ax6.plot(diff_epoch_time, diff2_kappa_perp_NLGC_plot / k0_perp, 'c', linestyle='-')
ax6.set_xlabel('Epoch (days)', fontsize=20)
ax6.set_ylabel('$\\kappa_\\perp$ ($\\times 10^{20}$ cm$^2$/s)', fontsize=20)
ax6.set_xlim(-4.0, 4.0)
ax6.set_ylim(2.0, 5.0)
ax6.axvline(0.0, color='r', linestyle='--')
ax6.tick_params(axis='x', labelsize=16)
ax6.tick_params(axis='y', labelsize=16)
ax6.text(2.7, 4.7, "NLGC", color='k', fontsize=20)

plt.show()
plt.close(fig)