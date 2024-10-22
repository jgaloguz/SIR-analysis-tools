# This code computes parallel and perpendicular diffusion coefficients according to multiple theories and the superposed epoch averaged results obtained in other codes.

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal as spsig
import sys
import os

# Running average with shrinking window size around edges of data
def edge_run_avg(mod_array, org_array, window_size):
   hws = window_size//2
   for i in range(hws,0,-1):
      mod_array[i] = np.mean(org_array[:i+hws])
      mod_array[-i] = np.mean(org_array[-i-hws:])

# conversion factors from data to cgs units
k_conv = 1.0e-5
Vmag_conv = 1.0e5
Bmag_conv = 1.0e-5
AU_conv = 1.496e13
c = 2.99792458e10
q = 4.8032047e-10
m = 1.67262192e-24
v = 1.7e10
B0 = 5.0e-5
csc = 1.0 / np.sin(3.0 * np.pi / 5.0)
k0_para = 1.0e22
k0_perp = 1.0e20

# Cyclotron frequency
def Omega(B):
   return q * B / (m * c)

# QLT parallel diffusion
def kappa_para_QLT(B, dB2_slab, l_b):
   return (3.0 * csc * v**3 * B**2) / (20.0 * Omega(B)**2 * l_b * dB2_slab) * (1.0 + (72.0 / 7.0) * (Omega(B) * l_b / v)**(5.0/3.0))

# Empirical parallel diffusion
def kappa_para_emp(k0, B):
   return k0 * (B0 / B)

# NLGC perpendicular diffusion
def kappa_perp_NLGC(B, dB2_2D, l_c, lamb_para):
   return (v / 3.0) * ((np.sqrt(3.0) / 3.0) * np.pi * 0.1189 * (dB2_2D / B**2) * l_c)**(2.0/3.0) * lamb_para**(1.0/3.0)

# LZP perpendicular diffusion
def kappa_perp_LZP(a, B, dB2_2D, kappa_para):
   return a * (dB2_2D / B**2) * kappa_para

# FLRW perpendicular diffusion
def kappa_perp_FLRW(folder, n_FD, n_intervals):
   kappa2_FL = np.zeros(n_intervals)
   SEA_counts = np.zeros(n_intervals)
# Iterate over events
   for FD_idx in range(n_FD):
# Iterate over subintervals
      for bin_idx in range(n_intervals):
         file_path = folder + "/k_spectrum_{:04d}_{:04d}.txt".format(FD_idx+1,bin_idx+1)
# Check that interval is valid (i.e. file was outputted)
         if os.path.isfile(file_path):
            SEA_counts[bin_idx] = SEA_counts[bin_idx] + 1
            k_spectrum_data = np.loadtxt(file_path)
            B0 = k_spectrum_data[0,0] # Units of B from file are nT
            k_range = k_spectrum_data[1:,0] * k_conv # Units of k from file are km^-1. Convert to cm^-1.
            k_spect = k_spectrum_data[1:,1] / k_conv # Units of PSD_k from file are nT^2 * km. Convert to nT^2 * cm.
            kappa2_FL[bin_idx] = kappa2_FL[bin_idx] + np.log((0.5 / B0**2) * sp.integrate.trapezoid(k_spect / np.square(k_range), k_range))
# Average over counts
   kappa2_FL = np.exp(np.divide(kappa2_FL, SEA_counts))
   return 0.5 * v * np.sqrt(kappa2_FL)

year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data

# Import SEA analysis results
SIR_macro_data = np.loadtxt("output/SEA_macro_" + str(year_start) + "-" + str(year_end) + ".lst")

SIR_turb_energy_data = np.loadtxt("output/SEA_turb_energy_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")

SIR_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")

SIR_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")

SIR_turb_decomp_alt_data = np.loadtxt("output/SEA_turb_decomp_alt_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")


epoch_time_macro = SIR_macro_data[:,0]
SIR_Bmag_avg = spsig.savgol_filter(SIR_macro_data[:,2] * Bmag_conv, 56, 0)

epoch_time_energy = SIR_turb_energy_data[:,0]
SIR_bfld_eng_avg = SIR_turb_energy_data[:,1]

epoch_time_micro = SIR_micro_data[:,0]
SIR_corr_length_avg = SIR_micro_data[:,2]
SIR_bend_length_avg = SIR_micro_data[:,3]

epoch_time_turb_decomp = SIR_turb_decomp_data[:,0]
SIR_P_slab_avg = SIR_turb_decomp_data[:,8]
SIR_P_twoD_avg = SIR_turb_decomp_data[:,9]

epoch_time_turb_decomp_alt = SIR_turb_decomp_alt_data[:,0]
SIR_C2_Cs_alt_avg = SIR_turb_decomp_alt_data[:,2]

n_intervals = np.size(epoch_time_energy)
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
n_FD = len(FD_list)
SIR_kappa_perp_FLRW_avg = kappa_perp_FLRW("output/k_spectra_" + str(year_start) + "-" + str(year_end), n_FD, n_intervals)

# For uniformity create interpolation functions and compute arrays of equal length
SIR_Bmag_func = sp.interpolate.interp1d(epoch_time_macro, SIR_Bmag_avg, bounds_error=False, fill_value=(SIR_Bmag_avg[0], SIR_Bmag_avg[-1]))
SIR_bfld_eng_func = sp.interpolate.interp1d(epoch_time_energy, SIR_bfld_eng_avg, bounds_error=False, fill_value=(SIR_bfld_eng_avg[0], SIR_bfld_eng_avg[-1]))
SIR_corr_length_func = sp.interpolate.interp1d(epoch_time_micro, SIR_corr_length_avg, bounds_error=False, fill_value=(SIR_corr_length_avg[0], SIR_corr_length_avg[-1]))
SIR_bend_length_func = sp.interpolate.interp1d(epoch_time_micro, SIR_bend_length_avg, bounds_error=False, fill_value=(SIR_bend_length_avg[0], SIR_bend_length_avg[-1]))
SIR_P_slab_func = sp.interpolate.interp1d(epoch_time_turb_decomp, SIR_P_slab_avg, bounds_error=False, fill_value=(SIR_P_slab_avg[0], SIR_P_slab_avg[-1]))
SIR_P_twoD_func = sp.interpolate.interp1d(epoch_time_turb_decomp, SIR_P_twoD_avg, bounds_error=False, fill_value=(SIR_P_twoD_avg[0], SIR_P_twoD_avg[-1]))
SIR_C2_Cs_alt_func = sp.interpolate.interp1d(epoch_time_turb_decomp_alt, SIR_C2_Cs_alt_avg, bounds_error=False, fill_value=(SIR_C2_Cs_alt_avg[0], SIR_C2_Cs_alt_avg[-1]))
SIR_kappa_perp_FLRW_func = sp.interpolate.interp1d(epoch_time_energy, SIR_kappa_perp_FLRW_avg, bounds_error=False, fill_value=(SIR_kappa_perp_FLRW_avg[0], SIR_kappa_perp_FLRW_avg[-1]))

n_pts = 8*24+1 # Number of epoch times to evaluate
epoch_time = np.linspace(-4.0, 4.0, num=n_pts)

SIR_Bmag = SIR_Bmag_func(epoch_time)
SIR_bfld_eng = SIR_bfld_eng_func(epoch_time)
SIR_corr_length = SIR_corr_length_func(epoch_time)
SIR_bend_length = SIR_bend_length_func(epoch_time)
SIR_P_slab = SIR_P_slab_func(epoch_time)
SIR_P_twoD = SIR_P_twoD_func(epoch_time)
SIR_pct_slab = spsig.savgol_filter(SIR_P_slab / (SIR_P_slab + SIR_P_twoD), 48, 0)
edge_run_avg(SIR_pct_slab, SIR_P_slab / (SIR_P_slab + SIR_P_twoD), 48)
SIR_pct_twoD = spsig.savgol_filter(SIR_P_twoD / (SIR_P_slab + SIR_P_twoD), 48, 0)
edge_run_avg(SIR_pct_twoD, SIR_P_twoD / (SIR_P_slab + SIR_P_twoD), 48)
SIR_pct_slab_alt = 1.0 / (1.0 + SIR_C2_Cs_alt_func(epoch_time))
SIR_pct_twoD_alt = 1.0 - SIR_pct_slab_alt
SIR_kappa_perp_FLRW_tot = SIR_kappa_perp_FLRW_func(epoch_time)

# Compute diffusion coefficients
SIR_kappa_para_emp = kappa_para_emp(4.0 * k0_para, SIR_Bmag)
SIR_lamb_para_emp = 3.0 * SIR_kappa_para_emp / v

a_LZP = 0.06 # Phenomenological constant
a_FLRW = 0.13 # Phenomenological constant
SIR_kappa_para_QLT = kappa_para_QLT(SIR_Bmag, SIR_bfld_eng * SIR_pct_slab, SIR_bend_length)
SIR_lamb_para_QLT = 3.0 * SIR_kappa_para_QLT / v
SIR_kappa_perp_LZP = kappa_perp_LZP(a_LZP, SIR_Bmag, SIR_bfld_eng * SIR_pct_twoD, SIR_kappa_para_QLT)
SIR_kappa_perp_NLGC = kappa_perp_NLGC(SIR_Bmag, SIR_bfld_eng * SIR_pct_twoD, SIR_corr_length, SIR_lamb_para_QLT)
SIR_kappa_perp_FLRW = a_FLRW * SIR_kappa_perp_FLRW_tot * SIR_pct_twoD

SIR_kappa_para_QLT_alt = kappa_para_QLT(SIR_Bmag, SIR_bfld_eng * SIR_pct_slab_alt, SIR_bend_length)
SIR_lamb_para_QLT_alt = 3.0 * SIR_kappa_para_QLT_alt / v
SIR_kappa_perp_LZP_alt = kappa_perp_LZP(a_LZP, SIR_Bmag, SIR_bfld_eng * SIR_pct_twoD_alt, SIR_kappa_para_QLT_alt)
SIR_kappa_perp_NLGC_alt = kappa_perp_NLGC(SIR_Bmag, SIR_bfld_eng * SIR_pct_twoD_alt, SIR_corr_length, SIR_lamb_para_QLT_alt)
SIR_kappa_perp_FLRW_alt = a_FLRW * SIR_kappa_perp_FLRW_tot * SIR_pct_twoD_alt

# Output results
file = open("output/SEA_diff_coeffs_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for pt in range(n_pts):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:12.4e}".format(SIR_kappa_para_emp[pt]))
   file.write("{:12.4e}{:12.4e}".format(SIR_kappa_para_QLT[pt],SIR_kappa_para_QLT_alt[pt]))
   file.write("{:12.4e}{:12.4e}".format(SIR_kappa_perp_LZP[pt],SIR_kappa_perp_LZP_alt[pt]))
   file.write("{:12.4e}{:12.4e}".format(SIR_kappa_perp_NLGC[pt],SIR_kappa_perp_NLGC_alt[pt]))
   file.write("{:12.4e}{:12.4e}\n".format(SIR_kappa_perp_FLRW[pt],SIR_kappa_perp_FLRW_alt[pt]))
file.close()
print("Analytic diffusion coefficients saved to disk.")
