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

# constants
c = 2.99792458e10
q = 4.8032047e-10
m = 1.67262192e-24
v = 1.7e10
B0 = 5.0e-5
csc = 1.0 / np.sin(3.0 * np.pi / 5.0)
k0_para = 1.0e22
k0_perp = 1.0e20

# numerical integration parameters
Nk = 0
Nkfit = 10
mu0 = 0.0001
Nmu = 10000
dmu = (1.0 - mu0) / (Nmu-1)
a_l = 0.0
b_l = 0.0
a_r = 0.0
b_r = 0.0

# Cyclotron frequency
def Omega(B):
   return q * B / (m * c)

# Empirical parallel diffusion
def kappa_para_KJ(k0, B):
   return k0 * (B0 / B)

# QLT parallel diffusion
def kappa_para_QLT_GJ(B, dB2_slab, l_b):
   return (3.0 * csc * v**3 * B**2) / (20.0 * Omega(B)**2 * l_b * dB2_slab) * (1.0 + (72.0 / 7.0) * (Omega(B) * l_b / v)**(5.0/3.0))

# NLGC perpendicular diffusion
def kappa_perp_NLGC(B, dB2_2D, l_c, lamb_para):
   return (v / 3.0) * ((np.sqrt(3.0) / 3.0) * np.pi * 0.1189 * (dB2_2D / B**2) * l_c)**(2.0/3.0) * lamb_para**(1.0/3.0)

# Fit spectrum
def fit_k_spect(k_range, k_spect):
   global Nk, a_l, b_l, a_r, b_r
   Nk = np.size(k_range)
# Left end
   lnk_range = np.log(k_range[:Nkfit])
   lnk_spect = np.log(k_spect[:Nkfit])
   params = np.polyfit(lnk_range, lnk_range, 1)
   a_l = params[0]
   b_l = params[1]
# Right end
   lnk_range = np.log(k_range[Nk-Nkfit:])
   lnk_spect = np.log(k_spect[Nk-Nkfit:])
   params = np.polyfit(lnk_range, lnk_range, 1)
   a_r = params[0]
   b_r = params[1]

# Slab spectrum
def g_slab(k_range, k_spect, k):
   k_idx = np.searchsorted(k_range, k)
   if(k_idx == 0):
      return np.exp(a_l + b_l*np.log(k))
   elif(k_idx == Nk):
      return np.exp(a_r + b_r*np.log(k))
   else:
      return k_spect[k_idx-1] + (k - k_range[k_idx-1]) * (k_spect[k_idx] - k_spect[k_idx-1]) / (k_range[k_idx] - k_range[k_idx-1]);

# QLT pitch-angle scattering coefficient
def D_mumu_QLT(B, k_range, k_spect, mu):
   k = Omega(B) / (v * mu);
   return 0.25 * np.pi * Omega(B) * (1.0 - mu**2) * k * g_slab(k_range, k_spect, k) / B**2;

# Integrate spectrum QLT
def integ_kappa_para_QLT(B, k_range, k_spect):
   mu = mu0
   S = 0.5 * (1.0 - mu**2)**2 / D_mumu_QLT(B, k_range, k_spect, mu)
   for i in range(1,Nmu-1):
      mu = mu + dmu
      S = S + (1.0 - mu**2)**2 / D_mumu_QLT(B, k_range, k_spect, mu)
   return 0.25 * v**2 * dmu * S

# Integrate spectrum UNLT
def integ_kappa_perp_UNLT(B, k_range, k_spect, kappa_para, kappa_perp_init):
   maxiter = 100
   eps = 1.0e-8
   kappa_perp_old = 0.0
   kappa_perp_new = kappa_perp_init
   for i in range(maxiter):
      kappa_perp_old = kappa_perp_new
      denom = 4.0 * kappa_perp_old * np.square(k_range) + (v**2 / kappa_para)
      kappa_perp_new = v**2 / (2.0 * B**2) * sp.integrate.trapezoid(k_spect / denom, k_range)
      if(np.abs(kappa_perp_new - kappa_perp_old) / (np.abs(kappa_perp_new) + np.abs(kappa_perp_old)) < eps):
         break

   if i == maxiter:
      print("\nWARNING: Maximum number of iterations reached in UNLT integration.\n")
   return kappa_perp_new

# QLT parallel diffusion and FLRW, LZP, UNLT perpendicular diffusion
def kappa_para_QLT_perp_FLRW_LZP_UNLT(folder, n_FD, n_intervals, pct_2D):
   kappa_para_QLT = np.zeros(n_intervals)
   kappa_perp_FLRW = np.zeros(n_intervals)
   kappa_perp_LZP = np.zeros(n_intervals)
   kappa_perp_UNLT = np.zeros(n_intervals)
   SEA_counts = np.zeros(n_intervals)
# Iterate over events
   for FD_idx in range(n_FD):
      print("Integrating intervals from event", FD_idx+1)
# Iterate over subintervals
      for bin_idx in range(n_intervals):
         file_path = folder + "/k_spectrum_{:04d}_{:04d}.txt".format(FD_idx+1,bin_idx+1)
# Check that interval is valid (i.e. file was outputted)
         if os.path.isfile(file_path):
            SEA_counts[bin_idx] = SEA_counts[bin_idx] + 1
            k_spectrum_data = np.loadtxt(file_path)
            B = k_spectrum_data[0,0] * Bmag_conv # Units of B from file are nT. Convert to G.
            k_range = k_spectrum_data[1:,0] * k_conv # Units of k from file are km^-1. Convert to cm^-1.
            k_spect = k_spectrum_data[1:,1] * Bmag_conv**2 / k_conv # Units of PSD_k from file are nT^2 * km. Convert to G^2 * cm.
            fit_k_spect(k_range,k_spect)
# Kappa parallel (QLT)
            kappa_para_QLT_bin = integ_kappa_para_QLT(B, k_range, (1.0 - pct_2D[bin_idx]) * k_spect)
            kappa_para_QLT[bin_idx] = kappa_para_QLT[bin_idx] + np.log(kappa_para_QLT_bin)
# Kappa perpendicular (FLRW)
            kappa2_FL = (0.5 / B**2) * sp.integrate.trapezoid(pct_2D[bin_idx] * k_spect / np.square(k_range), k_range)
            kappa_perp_FLRW_bin = 0.5 * v * np.sqrt(kappa2_FL)
            kappa_perp_FLRW[bin_idx] = kappa_perp_FLRW[bin_idx] + np.log(kappa_perp_FLRW_bin)
# Kappa perpendicular (LZP)
            kappa_perp_LZP_bin = (0.5 / B**2) * kappa_para_QLT_bin * sp.integrate.trapezoid(pct_2D[bin_idx] * k_spect, k_range)
            kappa_perp_LZP[bin_idx] = kappa_perp_LZP[bin_idx] + np.log(kappa_perp_LZP_bin)
# Kappa perpendicular (UNLT)
            lambda_para_QLT = 3 * kappa_para_QLT_bin / v
            kappa_perp_UNLT_bin = integ_kappa_perp_UNLT(B, k_range, pct_2D[bin_idx] * k_spect, lambda_para_QLT, kappa_perp_LZP_bin)
            kappa_perp_UNLT[bin_idx] = kappa_perp_UNLT[bin_idx] + np.log(kappa_perp_UNLT_bin)

# Average over counts
   kappa_para_QLT = np.exp(np.divide(kappa_para_QLT, SEA_counts))
   kappa_perp_FLRW = np.exp(np.divide(kappa_perp_FLRW, SEA_counts))
   kappa_perp_LZP = np.exp(np.divide(kappa_perp_LZP, SEA_counts))
   kappa_perp_UNLT = np.exp(np.divide(kappa_perp_UNLT, SEA_counts))
   return kappa_para_QLT, kappa_perp_FLRW, kappa_perp_LZP, kappa_perp_UNLT

year_start = int(sys.argv[1]) # Beginning year of clean data
year_end = year_start+3 # Final year of clean data

# Import SEA analysis results
SIR_macro_data = np.loadtxt("output/SEA_macro_" + str(year_start) + "-" + str(year_end) + ".lst")
SIR_turb_energy_data = np.loadtxt("output/SEA_turb_energy_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")
SIR_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")
SIR_turb_decomp_data = np.loadtxt("output/SEA_turb_decomp_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")

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

# For uniformity create interpolation functions and compute arrays of equal length
SIR_Bmag_func = sp.interpolate.interp1d(epoch_time_macro, SIR_Bmag_avg, bounds_error=False, fill_value=(SIR_Bmag_avg[0], SIR_Bmag_avg[-1]))
SIR_bfld_eng_func = sp.interpolate.interp1d(epoch_time_energy, SIR_bfld_eng_avg, bounds_error=False, fill_value=(SIR_bfld_eng_avg[0], SIR_bfld_eng_avg[-1]))
SIR_corr_length_func = sp.interpolate.interp1d(epoch_time_micro, SIR_corr_length_avg, bounds_error=False, fill_value=(SIR_corr_length_avg[0], SIR_corr_length_avg[-1]))
SIR_bend_length_func = sp.interpolate.interp1d(epoch_time_micro, SIR_bend_length_avg, bounds_error=False, fill_value=(SIR_bend_length_avg[0], SIR_bend_length_avg[-1]))
SIR_P_slab_func = sp.interpolate.interp1d(epoch_time_turb_decomp, SIR_P_slab_avg, bounds_error=False, fill_value=(SIR_P_slab_avg[0], SIR_P_slab_avg[-1]))
SIR_P_twoD_func = sp.interpolate.interp1d(epoch_time_turb_decomp, SIR_P_twoD_avg, bounds_error=False, fill_value=(SIR_P_twoD_avg[0], SIR_P_twoD_avg[-1]))

epoch_time = np.linspace(-4.0, 4.0, num=n_intervals)
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

# Compute diffusion coefficients
SIR_kappa_para_QLT, SIR_kappa_perp_FLRW, SIR_kappa_perp_LZP, SIR_kappa_perp_UNLT = \
   kappa_para_QLT_perp_FLRW_LZP_UNLT("output/k_spectra_" + str(year_start) + "-" + str(year_end), n_FD, n_intervals, SIR_pct_twoD)
SIR_kappa_para_KJ = kappa_para_KJ(4.0 * k0_para, SIR_Bmag)
SIR_kappa_para_GJ = kappa_para_QLT_GJ(SIR_Bmag, SIR_bfld_eng * SIR_pct_slab, SIR_bend_length)
SIR_lambda_para_GJ = 3.0 * SIR_kappa_para_GJ / v
SIR_kappa_perp_NLGC = kappa_perp_NLGC(SIR_Bmag, SIR_bfld_eng * SIR_pct_twoD, SIR_corr_length, SIR_lambda_para_GJ)

# Output results
file = open("output/SEA_diff_coeffs_" + str(year_start) + "-" + str(year_end) + ".lst", "w")
for pt in range(n_intervals):
   file.write("{:18.6f}".format(epoch_time[pt]))
   file.write("{:12.4e}".format(SIR_kappa_para_QLT[pt]))
   file.write("{:12.4e}".format(SIR_kappa_perp_FLRW[pt]))
   file.write("{:12.4e}".format(SIR_kappa_perp_LZP[pt]))
   file.write("{:12.4e}".format(SIR_kappa_perp_UNLT[pt]))
   file.write("{:12.4e}".format(SIR_kappa_para_KJ[pt]))
   file.write("{:12.4e}".format(SIR_kappa_para_GJ[pt]))
   file.write("{:12.4e}\n".format(SIR_kappa_perp_NLGC[pt]))
file.close()
print("Analytic diffusion coefficients saved to disk.")
