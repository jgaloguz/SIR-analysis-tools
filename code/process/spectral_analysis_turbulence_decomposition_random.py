# This code performs a random superposed epoch analysis of turbulence power and anisotropy in the solar wind and interplanetary magnetic field.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings
import sys

# conversion factors from data to cgs units
Vmag_conv = 1.0e5
Bmag_conv = 1.0e-5
m_p = 1.67262192e-24

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

# Pre-compute magnitude of magnetic field
Bmag = np.sqrt(SW_data[0]**2 + SW_data[1]**2 + SW_data[2]**2)

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
n_pts = 8*1350+1 # number of data points per event
half_n_pts = n_pts // 2 # half of n_pts
bin_full_width = 28+1 # number of data points in each subinterval
bin_half_width = bin_full_width // 2 # half of bin_full_width
bin_disp = 225 # number of data point shift between each subinterval
n_intervals = n_pts // bin_disp + 1 # number of (overlapping) subintervals
bin_pad = (n_pts % bin_disp) // 2 # number of datapoints to pad on first and last subinterval centers
missing_threshold = int(0.2 * bin_full_width) # threshold of number of missing datapoints above which interval is discarded 
para_threshold = np.deg2rad(-0.1) # threshold to discard parallel BV alignment
perp_threshold = np.deg2rad(90.1) # threshold to discard perpendicular BV alignment
slab_ang_threshold = np.deg2rad(20.0) # threshold to count data point as slab turbulence

sea_counts_list = np.zeros((n_intervals, n_seas)) # counts of samples in superposed epoch analysis
bfld_turb_eng_dens_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent magnetic field energy 
bmag_turb_eng_dens_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent magnetic field magnitude energy density
alfv_turb_eng_dens_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent Alfvenic energy density
vfld_turb_eng_dens_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent velocity energy 
P_xx_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent power in x direction of field aligned frame
P_yy_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent power in y direction of field aligned frame
P_zz_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged turbulent power in z direction of field aligned frame
sea_counts_slab_list = np.zeros((n_intervals, n_seas)) # counts of samples in superposed epoch analysis for slab component
sea_counts_twoD_list = np.zeros((n_intervals, n_seas)) # counts of samples in superposed epoch analysis for 2D component
P_slab_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged slab turbulent power
P_twoD_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged 2D turbulent power
theta_BV_avg_list = np.zeros((n_intervals, n_seas)) # superposed epoch averaged spectral index

# Iterate over SI list to find turbulent quantities, and output results to file
for sea_idx in range(n_seas):
   print("Random SEA", sea_idx)
   for epoch_list_idx in range(n_epochs):
      epoch_list[epoch_list_idx] = np.random.uniform(year_start+0.02,year_end+0.98) # Generate a new list of random times
   epoch_list.sort() # sort list of random times
   epoch_idx = 0 # zero epoch index
   for epoch_list_idx in range(n_epochs):
      while year_SW[epoch_idx] < epoch_list[epoch_list_idx]:
         epoch_idx = epoch_idx + 1
      Br_CIR = SW_data[0][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Bt_CIR = SW_data[1][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Bn_CIR = SW_data[2][epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
      Bm_CIR = Bmag[epoch_idx-half_n_pts-bin_half_width:epoch_idx+half_n_pts+bin_half_width+1]
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
         Bm_interval = Bm_CIR[bin_left:bin_right+1]
         Vr_interval = Vr_CIR[bin_left:bin_right+1]
         Vt_interval = Vt_CIR[bin_left:bin_right+1]
         Vn_interval = Vn_CIR[bin_left:bin_right+1]
         Np_interval = Np_CIR[bin_left:bin_right+1]

# convert data to field aligned frame
         B_mean = np.array([np.mean(Br_interval), np.mean(Bt_interval), np.mean(Bn_interval)])
         V_mean = np.array([np.mean(Vr_interval), np.mean(Vt_interval), np.mean(Vn_interval)])
         theta_BV = np.arccos(np.dot(B_mean, V_mean) / (np.linalg.norm(B_mean) * np.linalg.norm(V_mean)))
         if(theta_BV > 0.5 * np.pi):
            theta_BV = np.pi - theta_BV
         if(theta_BV < para_threshold):
            continue
         elif(theta_BV > perp_threshold):
            continue
         zhat = B_mean / np.linalg.norm(B_mean)
         V_perp = V_mean - zhat * np.dot(V_mean, zhat)
         xhat = V_perp / np.linalg.norm(V_perp)
         yhat = np.cross(zhat,xhat)
         
         Bx_faf = [xhat[0] * Br_interval[i] + xhat[1] * Bt_interval[i] + xhat[2] * Bn_interval[i] for i in range(bin_full_width)]
         By_faf = [yhat[0] * Br_interval[i] + yhat[1] * Bt_interval[i] + yhat[2] * Bn_interval[i] for i in range(bin_full_width)]
         Bz_faf = [zhat[0] * Br_interval[i] + zhat[1] * Bt_interval[i] + zhat[2] * Bn_interval[i] for i in range(bin_full_width)]

# Add to superposed epoch average
         B_var = np.var(Br_interval) + np.var(Bt_interval) + np.var(Bn_interval) # Variance of magnetic field
         Bm_var = np.var(Bm_interval) # Variance of magnitude of magnetic field
         V_var = np.var(Vr_interval) + np.var(Vt_interval) + np.var(Vn_interval) # Variance of velocity
         Np_avg = np.mean(Np_interval) # Average plasma density
         P_xx = np.var(Bx_faf)
         P_yy = np.var(By_faf)
         P_zz = np.var(Bz_faf)

         sea_counts_list[bin_idx,sea_idx] = sea_counts_list[bin_idx,sea_idx] + 1.0
         bfld_turb_eng_dens_avg_list[bin_idx, sea_idx] = bfld_turb_eng_dens_avg_list[bin_idx, sea_idx] + (Bmag_conv**2) * B_var
         bmag_turb_eng_dens_avg_list[bin_idx, sea_idx] = bmag_turb_eng_dens_avg_list[bin_idx, sea_idx] + (Bmag_conv**2) * Bm_var
         alfv_turb_eng_dens_avg_list[bin_idx, sea_idx] = alfv_turb_eng_dens_avg_list[bin_idx, sea_idx] + (Bmag_conv**2) * B_var / (4.0 * np.pi * Np_avg * m_p)
         vfld_turb_eng_dens_avg_list[bin_idx, sea_idx] = vfld_turb_eng_dens_avg_list[bin_idx, sea_idx] + (Vmag_conv**2) * V_var
         P_xx_avg_list[bin_idx,sea_idx] = P_xx_avg_list[bin_idx,sea_idx] + P_xx
         P_yy_avg_list[bin_idx,sea_idx] = P_yy_avg_list[bin_idx,sea_idx] + P_yy
         P_zz_avg_list[bin_idx,sea_idx] = P_zz_avg_list[bin_idx,sea_idx] + P_zz
         if(theta_BV < slab_ang_threshold):
            P_slab_avg_list[bin_idx,sea_idx] = P_slab_avg_list[bin_idx,sea_idx] + (P_xx + P_yy)
            sea_counts_slab_list[bin_idx,sea_idx] = sea_counts_slab_list[bin_idx,sea_idx] + 1.0
         elif(theta_BV > 0.5 * np.pi - slab_ang_threshold):
            P_twoD_avg_list[bin_idx,sea_idx] = P_twoD_avg_list[bin_idx,sea_idx] + (P_xx + P_yy)
            sea_counts_twoD_list[bin_idx,sea_idx] = sea_counts_twoD_list[bin_idx,sea_idx] + 1.0
         theta_BV_avg_list[bin_idx,sea_idx] = theta_BV_avg_list[bin_idx,sea_idx] + theta_BV

# Normalize by number of samples
   bfld_turb_eng_dens_avg_list[:,sea_idx] = np.divide(bfld_turb_eng_dens_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   bmag_turb_eng_dens_avg_list[:,sea_idx] = np.divide(bmag_turb_eng_dens_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   alfv_turb_eng_dens_avg_list[:,sea_idx] = np.divide(alfv_turb_eng_dens_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   vfld_turb_eng_dens_avg_list[:,sea_idx] = np.divide(vfld_turb_eng_dens_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   P_xx_avg_list[:,sea_idx] = np.divide(P_xx_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   P_yy_avg_list[:,sea_idx] = np.divide(P_yy_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   P_zz_avg_list[:,sea_idx] = np.divide(P_zz_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])
   P_slab_avg_list[:,sea_idx] = np.divide(P_slab_avg_list[:,sea_idx],sea_counts_slab_list[:,sea_idx])
   P_twoD_avg_list[:,sea_idx] = np.divide(P_twoD_avg_list[:,sea_idx],sea_counts_twoD_list[:,sea_idx])
   theta_BV_avg_list[:,sea_idx] = np.divide(theta_BV_avg_list[:,sea_idx],sea_counts_list[:,sea_idx])

# Find averages and standard deviations
bfld_turb_eng_dens_avg = np.nanmean(bfld_turb_eng_dens_avg_list, axis=1) # average
bmag_turb_eng_dens_avg = np.nanmean(bmag_turb_eng_dens_avg_list, axis=1) # average
alfv_turb_eng_dens_avg = np.nanmean(alfv_turb_eng_dens_avg_list, axis=1) # average
vfld_turb_eng_dens_avg = np.nanmean(vfld_turb_eng_dens_avg_list, axis=1) # average
P_xx_avg = np.nanmean(P_xx_avg_list, axis=1) # average
P_yy_avg = np.nanmean(P_yy_avg_list, axis=1) # average
P_zz_avg = np.nanmean(P_zz_avg_list, axis=1) # average
P_slab_avg = np.nanmean(P_slab_avg_list, axis=1) # average
P_twoD_avg = np.nanmean(P_twoD_avg_list, axis=1) # average
theta_BV_avg = np.nanmean(theta_BV_avg_list, axis=1) # average
bfld_turb_eng_dens_std = np.nanstd(bfld_turb_eng_dens_avg_list, axis=1) # standard deviation
bmag_turb_eng_dens_std = np.nanstd(bmag_turb_eng_dens_avg_list, axis=1) # standard deviation
alfv_turb_eng_dens_std = np.nanstd(alfv_turb_eng_dens_avg_list, axis=1) # standard deviation
vfld_turb_eng_dens_std = np.nanstd(vfld_turb_eng_dens_avg_list, axis=1) # standard deviation
P_xx_std = np.nanstd(P_xx_avg_list, axis=1) # standard deviation
P_yy_std = np.nanstd(P_yy_avg_list, axis=1) # standard deviation
P_zz_std = np.nanstd(P_zz_avg_list, axis=1) # standard deviation
P_slab_std = np.nanstd(P_slab_avg_list, axis=1) # standard deviation
P_twoD_std = np.nanstd(P_twoD_avg_list, axis=1) # standard deviation
theta_BV_std = np.nanstd(theta_BV_avg_list, axis=1) # standard deviation

# Plot
init_epoch = -4.0+(bin_half_width/1350)
final_epoch = init_epoch + (n_intervals-1)*(bin_disp/1350)
epoch_time = np.linspace(init_epoch,final_epoch, num=n_intervals)
fig = plt.figure(figsize=(12, 10), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)

ax1 = fig.add_subplot(221, projection='rectilinear')
ax1.plot(epoch_time, alfv_turb_eng_dens_avg)
ax1.plot(epoch_time, alfv_turb_eng_dens_avg+alfv_turb_eng_dens_std)
ax1.plot(epoch_time, alfv_turb_eng_dens_avg-alfv_turb_eng_dens_std)
ax1.plot(epoch_time, vfld_turb_eng_dens_avg)
ax1.plot(epoch_time, vfld_turb_eng_dens_avg+vfld_turb_eng_dens_std)
ax1.plot(epoch_time, vfld_turb_eng_dens_avg-vfld_turb_eng_dens_std)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Turbulent Energy Density', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')
ax2.plot(epoch_time, P_twoD_avg / P_slab_avg)
ratio_std = (P_twoD_avg / P_slab_avg) * np.sqrt((P_twoD_std/P_twoD_avg)**2 + (P_slab_std/P_slab_avg)**2)
ax2.plot(epoch_time, (P_twoD_avg / P_slab_avg)+ratio_std)
ax2.plot(epoch_time, (P_twoD_avg / P_slab_avg)-ratio_std)
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('$P_2 / P_s$', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.axvline(0.0, color='k')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')
ax3.plot(epoch_time, P_zz_avg / (P_yy_avg + P_xx_avg))
ratio_std = (P_zz_avg / (P_yy_avg + P_xx_avg)) * np.sqrt((P_zz_std/P_zz_avg)**2 + ((P_yy_std + P_xx_std)/(P_yy_avg + P_xx_avg))**2)
ax3.plot(epoch_time, (P_zz_avg / (P_yy_avg + P_xx_avg))+ratio_std)
ax3.plot(epoch_time, (P_zz_avg / (P_yy_avg + P_xx_avg))-ratio_std)
ax3.plot(epoch_time, 3.0 * bmag_turb_eng_dens_avg / bfld_turb_eng_dens_avg)
ratio_std = 3.0 * (bmag_turb_eng_dens_avg / bfld_turb_eng_dens_avg) * np.sqrt((bmag_turb_eng_dens_std/bmag_turb_eng_dens_avg)**2 + (bfld_turb_eng_dens_std/bfld_turb_eng_dens_avg)**2)
ax3.plot(epoch_time, 3.0 * (bmag_turb_eng_dens_avg / bfld_turb_eng_dens_avg)+ratio_std)
ax3.plot(epoch_time, 3.0 * (bmag_turb_eng_dens_avg / bfld_turb_eng_dens_avg)-ratio_std)
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('$P_{zz} / (P_{xx} + P_{yy})$', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='k')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')
ax4.plot(epoch_time, theta_BV_avg)
ax4.plot(epoch_time, theta_BV_avg+theta_BV_std)
ax4.plot(epoch_time, theta_BV_avg-theta_BV_std)
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('Theta BV', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='k')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
file = open("output/SEA_turb_decomp_" + str(year_start) + "-" + str(year_end) + "_random.lst", "w")
for bin_idx in range(n_intervals):
   file.write("{:18.6f}".format(epoch_time[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}".format(bfld_turb_eng_dens_avg[bin_idx], bfld_turb_eng_dens_std[bin_idx],
                                                        bmag_turb_eng_dens_avg[bin_idx], bmag_turb_eng_dens_std[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}".format(alfv_turb_eng_dens_avg[bin_idx], alfv_turb_eng_dens_std[bin_idx],
                                                        vfld_turb_eng_dens_avg[bin_idx], vfld_turb_eng_dens_std[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}".format(P_xx_avg[bin_idx], P_xx_std[bin_idx],
                                                                        P_yy_avg[bin_idx], P_yy_std[bin_idx],
                                                                        P_zz_avg[bin_idx], P_zz_std[bin_idx]))

   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}\n".format(P_slab_avg[bin_idx], P_slab_std[bin_idx],
                                                                          P_twoD_avg[bin_idx], P_twoD_std[bin_idx],
                                                                          theta_BV_avg[bin_idx], theta_BV_std[bin_idx]))
file.close()
print("Superposed epoch analysis results for turbulent quantities saved to disk.")