# This code plots sample fits of a turbulence geometry decomposition method used to distinguish between slab and 2D modes in the interplanetary magnetic field.

import matplotlib.pyplot as plt
import scipy.signal as spsig
import numpy as np

# Ratio of 2D to slab amplitude
def Pyy_Pxx(q, theta, C2_Cs):
   costq = np.cos(theta)**(q-1.0)
   sintq = np.sin(theta)**(q-1.0)
   Pxx_slab = (q + 1.0) * costq
   Pxx_twoD = 2.0 * C2_Cs * sintq
   Pyy_slab = Pxx_slab
   Pyy_twoD = q * Pxx_twoD
   return (Pyy_slab + Pyy_twoD) / (Pxx_slab + Pxx_twoD)

# Import SEA analysis results
SIR1_turb_decomp_alt_data = np.loadtxt("output/SEA_turb_decomp_alt_without_SBs_2007-2010.lst")
SIR2_turb_decomp_alt_data = np.loadtxt("output/SEA_turb_decomp_alt_without_SBs_2018-2021.lst")
SIR1_turb_decomp_alt_fits_data = np.loadtxt("output/SEA_turb_decomp_alt_fits_without_SBs_2007-2010.lst")
SIR2_turb_decomp_alt_fits_data = np.loadtxt("output/SEA_turb_decomp_alt_fits_without_SBs_2018-2021.lst")

# Define data
theta_fit = np.linspace(0.0, 0.5*np.pi, num=200)
SIR_theta = SIR1_turb_decomp_alt_fits_data[:,0]
SIR1_q = SIR1_turb_decomp_alt_data[:,1]
SIR1_C2_Cs = SIR1_turb_decomp_alt_data[:,2]
n_segs = np.size(SIR1_q)
SIR1_P_yy_P_xx_seg_avg = [[] for seg in range(n_segs)]
SIR1_P_yy_P_xx_seg_std = [[] for seg in range(n_segs)]
for seg in range(n_segs):
   SIR1_P_yy_P_xx_seg_avg[seg] = SIR1_turb_decomp_alt_fits_data[:,2*seg+1]
   SIR1_P_yy_P_xx_seg_std[seg] = SIR1_turb_decomp_alt_fits_data[:,2*seg+2]
SIR2_q = SIR2_turb_decomp_alt_data[:,1]
SIR2_C2_Cs = SIR2_turb_decomp_alt_data[:,2]
SIR2_P_yy_P_xx_seg_avg = [[] for seg in range(n_segs)]
SIR2_P_yy_P_xx_seg_std = [[] for seg in range(n_segs)]
for seg in range(n_segs):
   SIR2_P_yy_P_xx_seg_avg[seg] = SIR2_turb_decomp_alt_fits_data[:,2*seg+1]
   SIR2_P_yy_P_xx_seg_std[seg] = SIR2_turb_decomp_alt_fits_data[:,2*seg+2]

# Plot
fig = plt.figure(figsize=(12, 8), layout='tight')
# fig.suptitle('Alternative Turbulence Decomposition Superposed Epoch Analysis', fontsize=24)
ax1 = fig.add_subplot(221, projection='rectilinear')

plot_idx = int(0.5 * (n_segs / 4))
ax1.errorbar(np.rad2deg(SIR_theta), SIR1_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR1_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='b', capsize=5)
ax1.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR1_q[plot_idx], theta_fit, SIR1_C2_Cs[plot_idx]), color='b', linestyle='-')
ax1.axhline(SIR1_q[plot_idx], color='b', linestyle='--')
ax1.errorbar(np.rad2deg(SIR_theta), SIR2_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR2_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='c', capsize=5)
ax1.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR2_q[plot_idx], theta_fit, SIR2_C2_Cs[plot_idx]), color='c', linestyle='-')
ax1.axhline(SIR2_q[plot_idx], color='c', linestyle='--')
ax1.text(76.0, 0.86, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR1_C2_Cs[plot_idx])), color='b', fontsize=16)
ax1.text(76.0, 0.76, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR2_C2_Cs[plot_idx])), color='c', fontsize=16)
ax1.axhline(1.0, color='k', linestyle='--')
ax1.set_xlabel('$\\theta_{\\mathrm{BV}}$ ($^\\circ$)', fontsize=20)
ax1.set_ylabel('$P_{yy}/P_{xx}$', fontsize=20)
ax1.set_xlim(0.0, 90.0)
ax1.set_ylim(0.7, 2.0)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')

plot_idx = int(1.5 * (n_segs / 4))
ax2.errorbar(np.rad2deg(SIR_theta), SIR1_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR1_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='b', capsize=5)
ax2.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR1_q[plot_idx], theta_fit, SIR1_C2_Cs[plot_idx]), color='b', linestyle='-')
ax2.axhline(SIR1_q[plot_idx], color='b', linestyle='--')
ax2.errorbar(np.rad2deg(SIR_theta), SIR2_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR2_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='c', capsize=5)
ax2.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR2_q[plot_idx], theta_fit, SIR2_C2_Cs[plot_idx]), color='c', linestyle='-')
ax2.axhline(SIR2_q[plot_idx], color='c', linestyle='--')
ax2.text(77.0, 0.87, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR1_C2_Cs[plot_idx])), color='b', fontsize=16)
ax2.text(77.0, 0.77, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR2_C2_Cs[plot_idx])), color='c', fontsize=16)
ax2.axhline(1.0, color='k', linestyle='--')
ax2.set_xlabel('$\\theta_{\\mathrm{BV}}$ ($^\\circ$)', fontsize=20)
ax2.set_ylabel('$P_{yy}/P_{xx}$', fontsize=20)
ax2.set_xlim(0.0, 90.0)
ax2.set_ylim(0.7, 2.0)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')

plot_idx = int(2.5 * (n_segs / 4))
ax3.errorbar(np.rad2deg(SIR_theta), SIR1_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR1_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='b', capsize=5)
ax3.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR1_q[plot_idx], theta_fit, SIR1_C2_Cs[plot_idx]), color='b', linestyle='-')
ax3.axhline(SIR1_q[plot_idx], color='b', linestyle='--')
ax3.errorbar(np.rad2deg(SIR_theta), SIR2_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR2_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='c', capsize=5)
ax3.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR2_q[plot_idx], theta_fit, SIR2_C2_Cs[plot_idx]), color='c', linestyle='-')
ax3.axhline(SIR2_q[plot_idx], color='c', linestyle='--')
ax3.text(77.0, 0.87, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR1_C2_Cs[plot_idx])), color='b', fontsize=16)
ax3.text(77.0, 0.77, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR2_C2_Cs[plot_idx])), color='c', fontsize=16)
ax3.axhline(1.0, color='k', linestyle='--')
ax3.set_xlabel('$\\theta_{\\mathrm{BV}}$ ($^\\circ$)', fontsize=20)
ax3.set_ylabel('$P_{yy}/P_{xx}$', fontsize=20)
ax3.set_xlim(0.0, 90.0)
ax3.set_ylim(0.7, 2.0)
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')

plot_idx = int(3.5 * (n_segs / 4))
ax4.errorbar(np.rad2deg(SIR_theta), SIR1_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR1_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='b', capsize=5)
ax4.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR1_q[plot_idx], theta_fit, SIR1_C2_Cs[plot_idx]), color='b', linestyle='-')
ax4.axhline(SIR1_q[plot_idx], color='b', linestyle='--')
ax4.errorbar(np.rad2deg(SIR_theta), SIR2_P_yy_P_xx_seg_avg[plot_idx], yerr=SIR2_P_yy_P_xx_seg_std[plot_idx], fmt='s', c='c', capsize=5)
ax4.plot(np.rad2deg(theta_fit), Pyy_Pxx(SIR2_q[plot_idx], theta_fit, SIR2_C2_Cs[plot_idx]), color='c', linestyle='-')
ax4.axhline(SIR2_q[plot_idx], color='c', linestyle='--')
ax4.text(77.0, 0.87, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR1_C2_Cs[plot_idx])), color='b', fontsize=16)
ax4.text(77.0, 0.77, "$r_s$ = {:.2f}".format(1.0 / (1.0 + SIR2_C2_Cs[plot_idx])), color='c', fontsize=16)
ax4.axhline(1.0, color='k', linestyle='--')
ax4.set_xlabel('$\\theta_{\\mathrm{BV}}$ ($^\\circ$)', fontsize=20)
ax4.set_ylabel('$P_{yy}/P_{xx}$', fontsize=20)
ax4.set_xlim(0.0, 90.0)
ax4.set_ylim(0.7, 2.0)
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)