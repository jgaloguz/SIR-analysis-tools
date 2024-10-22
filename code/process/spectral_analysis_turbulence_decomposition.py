# This code performs a superposed epoch analysis of turbulence power and anisotropy in the solar wind and interplanetary magnetic field for a list of events.

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import warnings
import sys

# conversion factors from data to cgs units
Vmag_conv = 1.0e5
Bmag_conv = 1.0e-5
m_p = 1.67262192e-24

# Flag to remove (or not) SB
remove_SB = True

# Function to fill in missing values of data by interpolating bounding valid data
def Imputate(time_array, data_array):
   N = np.size(data_array)
# Fix first value if necessary by setting it equal to the first non-zero value
   i = 0
   while not np.isfinite(data_array[i]):
      i = i + 1
   data_array[0] = data_array[i]
# Fix last value if necessary by setting it equal to the last non-zero value
   i = N-1
   while not np.isfinite(data_array[i]):
      i = i - 1
   data_array[N-1] = data_array[i]
# Interpolate missing_values between first and last
   i = 1
   while i < N-1:
# If value is missing
      if not np.isfinite(data_array[i]):
# Get last value not missing (just previous value)
         k1 = i-1
         j = i+1
         while not np.isfinite(data_array[j]):
            j = j + 1
# Get next value not missing (guaranteed to exist)
         k2 = j
# Interpolate
         for k in range(k1+1,k2):
            data_array[k] = (data_array[k1] * (time_array[k2] - time_array[k]) + data_array[k2] * (time_array[k] - time_array[k1])) / (time_array[k2] - time_array[k1])
         i = j
      i = i + 1

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

# Import list of stream interfaces (SI)
FD_list = [] # List of SI
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

# Import list of sector boundaries (SB)
SB_list = [[] for _ in range(n_FD)] # List of SB indices
if remove_SB:
   file = open("output/SB_near_events_" + str(year_start) + "-" + str(year_end) + ".lst", "r")
   for FD_idx in range(n_FD):
      center_idx_strs = file.readline().split()
      SB_list[FD_idx] = [int(idx_str) for idx_str in center_idx_strs]
   file.close()

# Define parameters for turbulent analysis
n_pts = 8*1350+1 # number of data points per CIR event (each centered at SI)
half_n_pts = n_pts // 2 # half of n_pts
bin_full_width = 28+1 # number of data points in each subinterval
bin_half_width = bin_full_width // 2 # half of bin_full_width
bin_disp = 28+1 # number of data point shift between each subinterval
n_intervals = n_pts // bin_disp + 1 # number of (overlapping) subintervals
bin_pad = (n_pts % bin_disp) // 2 # number of datapoints to pad on first and last subinterval centers
missing_threshold = int(0.2 * bin_full_width) # threshold of number of missing datapoints above which interval is discarded
para_threshold = np.deg2rad(-0.1) # threshold to discard parallel BV alignment
perp_threshold = np.deg2rad(90.1) # threshold to discard perpendicular BV alignment
slab_ang_threshold = np.deg2rad(20.0) # threshold to count data point as slab turbulence

sea_counts = np.zeros(n_intervals) # counts of samples in superposed epoch analysis
bfld_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent magnetic field energy density
bmag_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent magnetic field magnitude energy density
alfv_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent Alfvenic energy density
vfld_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent velocity energy 
P_xx_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent power in x direction of field aligned frame
P_yy_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent power in y direction of field aligned frame
P_zz_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent power in z direction of field aligned frame
sea_counts_slab = np.zeros(n_intervals) # counts of samples in superposed epoch analysis for slab component
sea_counts_twoD = np.zeros(n_intervals) # counts of samples in superposed epoch analysis for 2D component
P_slab_avg = np.zeros(n_intervals) # superposed epoch averaged slab turbulent power
P_twoD_avg = np.zeros(n_intervals) # superposed epoch averaged 2D turbulent power
theta_BV_avg = np.zeros(n_intervals) # superposed epoch averaged spectral index

epoch_idx = 0 # zero epoch index
# Iterate over SI list to find turbulent quantities, and output results to file
for FD_idx in range(n_FD):
   print("Analyzing CIR", FD_idx+1)
   while year_SW[epoch_idx] < FD_list[FD_idx]:
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
# check overlap with any SB
      SB_overlap = False
      for SB_idx in SB_list[FD_idx]:
         if bin_left <= SB_idx + bin_half_width <= bin_right:
            SB_overlap = True
            break
      if not SB_overlap:
# check missing data threshold
         if np.sum(B_missing_CIR[bin_left:bin_right+1] > missing_threshold):
            continue
         Br_interval = Br_CIR[bin_left:bin_right+1]
         Bt_interval = Bt_CIR[bin_left:bin_right+1]
         Bn_interval = Bn_CIR[bin_left:bin_right+1]
         Bm_interval = Bm_CIR[bin_left:bin_right+1]
         Vr_interval = Vr_CIR[bin_left:bin_right+1]
         Vt_interval = Vt_CIR[bin_left:bin_right+1]
         Vn_interval = Vn_CIR[bin_left:bin_right+1]
         Np_interval = Np_CIR[bin_left:bin_right+1]

# convert data to field aligned frame
         Br_mean = np.mean(Br_interval)
         Bt_mean = np.mean(Bt_interval)
         Bn_mean = np.mean(Bn_interval)
         Vr_mean = np.mean(Vr_interval)
         Vt_mean = np.mean(Vt_interval)
         Vn_mean = np.mean(Vn_interval)

         B_mean = np.array([Br_mean, Bt_mean, Bn_mean])
         V_mean = np.array([Vr_mean, Vt_mean, Vn_mean])
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

         sea_counts[bin_idx] = sea_counts[bin_idx] + 1.0
         bfld_turb_eng_dens_avg[bin_idx] = bfld_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * B_var
         bmag_turb_eng_dens_avg[bin_idx] = bmag_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * Bm_var
         alfv_turb_eng_dens_avg[bin_idx] = alfv_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * B_var / (4.0 * np.pi * Np_avg * m_p)
         vfld_turb_eng_dens_avg[bin_idx] = vfld_turb_eng_dens_avg[bin_idx] + (Vmag_conv**2) * V_var
         P_xx_avg[bin_idx] = P_xx_avg[bin_idx] + P_xx
         P_yy_avg[bin_idx] = P_yy_avg[bin_idx] + P_yy
         P_zz_avg[bin_idx] = P_zz_avg[bin_idx] + P_zz
         if(theta_BV < slab_ang_threshold):
            P_slab_avg[bin_idx] = P_slab_avg[bin_idx] + (P_xx + P_yy)
            sea_counts_slab[bin_idx] = sea_counts_slab[bin_idx] + 1.0
         elif(theta_BV > 0.5 * np.pi - slab_ang_threshold):
            P_twoD_avg[bin_idx] = P_twoD_avg[bin_idx] + (P_xx + P_yy)
            sea_counts_twoD[bin_idx] = sea_counts_twoD[bin_idx] + 1.0
         theta_BV_avg[bin_idx] = theta_BV_avg[bin_idx] + theta_BV

# Normalize by number of 
bfld_turb_eng_dens_avg = np.divide(bfld_turb_eng_dens_avg,sea_counts)
bmag_turb_eng_dens_avg = np.divide(bmag_turb_eng_dens_avg,sea_counts)
alfv_turb_eng_dens_avg = np.divide(alfv_turb_eng_dens_avg,sea_counts)
vfld_turb_eng_dens_avg = np.divide(vfld_turb_eng_dens_avg,sea_counts)
P_xx_avg = np.divide(P_xx_avg,sea_counts)
P_yy_avg = np.divide(P_yy_avg,sea_counts)
P_zz_avg = np.divide(P_zz_avg,sea_counts)
theta_BV_avg = np.divide(theta_BV_avg,sea_counts)

P_slab_avg = np.divide(P_slab_avg,sea_counts_slab)
P_twoD_avg = np.divide(P_twoD_avg,sea_counts_twoD)

# Plot
init_epoch = -4.0+(bin_pad/1350)
final_epoch = init_epoch + (n_intervals-1)*(bin_disp/1350)
epoch_time = np.linspace(init_epoch,final_epoch, num=n_intervals)
fig = plt.figure(figsize=(12, 10), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)

ax1 = fig.add_subplot(221, projection='rectilinear')
ax1.plot(epoch_time, alfv_turb_eng_dens_avg)
ax1.plot(epoch_time, vfld_turb_eng_dens_avg)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Turbulent Energy Density', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(222, projection='rectilinear')
Imputate(epoch_time, P_slab_avg)
Imputate(epoch_time, P_twoD_avg)
pct_2D = 100 * P_twoD_avg / (P_slab_avg + P_twoD_avg)
pct_2D_avg = spsig.savgol_filter(pct_2D, 48, 0)
ax2.plot(epoch_time, pct_2D)
ax2.plot(epoch_time, pct_2D_avg)
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('% 2D', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.set_ylim(0.0,100.0)
ax2.axvline(0.0, color='k')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax3 = fig.add_subplot(223, projection='rectilinear')
ax3.plot(epoch_time, P_zz_avg / (P_yy_avg + P_xx_avg))
ax3.plot(epoch_time, 3.0 * bmag_turb_eng_dens_avg / bfld_turb_eng_dens_avg)
ax3.set_xlabel('Epoch (days)', fontsize=20)
ax3.set_ylabel('$P_{\\parallel} / P_{\\perp}$', fontsize=20)
ax3.set_xlim(-4.0, 4.0)
ax3.axvline(0.0, color='k')
ax3.tick_params(axis='x', labelsize=16)
ax3.tick_params(axis='y', labelsize=16)

ax4 = fig.add_subplot(224, projection='rectilinear')
ax4.plot(epoch_time, np.rad2deg(theta_BV_avg))
ax4.set_xlabel('Epoch (days)', fontsize=20)
ax4.set_ylabel('$\\theta_{BV}$', fontsize=20)
ax4.set_xlim(-4.0, 4.0)
ax4.axvline(0.0, color='k')
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
if remove_SB:
   SB_label = "without_SBs_"
else:
   SB_label = "with_SBs_"
file = open("output/SEA_turb_decomp_" + SB_label + str(year_start) + "-" + str(year_end) + ".lst", "w")
for bin_idx in range(n_intervals):
   file.write("{:18.6f}".format(epoch_time[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}".format(bfld_turb_eng_dens_avg[bin_idx], bmag_turb_eng_dens_avg[bin_idx],
                                                        alfv_turb_eng_dens_avg[bin_idx], vfld_turb_eng_dens_avg[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}{:12.3e}\n".format(P_xx_avg[bin_idx], P_yy_avg[bin_idx], P_zz_avg[bin_idx],
                                                                          P_slab_avg[bin_idx], P_twoD_avg[bin_idx], theta_BV_avg[bin_idx]))
file.close()
print("Superposed epoch analysis results for turbulent quantities saved to disk.")