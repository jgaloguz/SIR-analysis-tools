# This code performs a superposed epoch analysis of turbulence power in the solar wind and interplanetary magnetic field for a list of events.

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
bin_full_width = 56+1 # number of data points in each subinterval
bin_half_width = bin_full_width // 2 # half of bin_full_width
bin_disp = 56 # number of data point shift between each subinterval
n_intervals = n_pts // bin_disp + 1 # number of (overlapping) subintervals
bin_pad = (n_pts % bin_disp) // 2 # number of datapoints to pad on first and last subinterval centers
missing_threshold = int(0.2 * bin_full_width) # threshold of number of missing datapoints above which interval is discarded

sea_counts = np.zeros(n_intervals) # counts of samples in superposed epoch analysis
bfld_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent magnetic field energy density
bmag_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent magnetic field magnitude energy density
alfv_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent Alfvenic energy density
vfld_turb_eng_dens_avg = np.zeros(n_intervals) # superposed epoch averaged turbulent velocity energy 

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

# Add to superposed epoch average
         B_var = np.var(Br_interval) + np.var(Bt_interval) + np.var(Bn_interval) # Variance of magnetic field
         Bm_var = np.var(Bm_interval) # Variance of magnitude of magnetic field
         V_var = np.var(Vr_interval) + np.var(Vt_interval) + np.var(Vn_interval) # Variance of velocity
         Np_avg = np.mean(Np_interval) # Average plasma density

         sea_counts[bin_idx] = sea_counts[bin_idx] + 1.0
         bfld_turb_eng_dens_avg[bin_idx] = bfld_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * B_var
         bmag_turb_eng_dens_avg[bin_idx] = bmag_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * Bm_var
         alfv_turb_eng_dens_avg[bin_idx] = alfv_turb_eng_dens_avg[bin_idx] + (Bmag_conv**2) * B_var / (4.0 * np.pi * Np_avg * m_p)
         vfld_turb_eng_dens_avg[bin_idx] = vfld_turb_eng_dens_avg[bin_idx] + (Vmag_conv**2) * V_var

# Normalize by number of 
bfld_turb_eng_dens_avg = np.divide(bfld_turb_eng_dens_avg,sea_counts)
bmag_turb_eng_dens_avg = np.divide(bmag_turb_eng_dens_avg,sea_counts)
alfv_turb_eng_dens_avg = np.divide(alfv_turb_eng_dens_avg,sea_counts)
vfld_turb_eng_dens_avg = np.divide(vfld_turb_eng_dens_avg,sea_counts)

# Plot
init_epoch = -4.0+(bin_pad/1350)
final_epoch = init_epoch + (n_intervals-1)*(bin_disp/1350)
epoch_time = np.linspace(init_epoch,final_epoch, num=n_intervals)
fig = plt.figure(figsize=(12, 10), layout='tight')
fig.suptitle('Superposed Epoch Analysis', fontsize=24)

ax1 = fig.add_subplot(211, projection='rectilinear')
ax1.plot(epoch_time, bfld_turb_eng_dens_avg)
ax1.plot(epoch_time, bmag_turb_eng_dens_avg)
ax1.set_xlabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('Turbulent Energy Density', fontsize=20)
ax1.set_xlim(-4.0, 4.0)
ax1.axvline(0.0, color='k')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax2 = fig.add_subplot(212, projection='rectilinear')
ax2.plot(epoch_time, alfv_turb_eng_dens_avg)
ax2.plot(epoch_time, vfld_turb_eng_dens_avg)
ax2.set_xlabel('Epoch (days)', fontsize=20)
ax2.set_ylabel('Turbulent Energy Density', fontsize=20)
ax2.set_xlim(-4.0, 4.0)
ax2.axvline(0.0, color='k')
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

# Output results
if remove_SB:
   SB_label = "without_SBs_"
else:
   SB_label = "with_SBs_"
file = open("output/SEA_turb_energy_" + SB_label + str(year_start) + "-" + str(year_end) + ".lst", "w")
for bin_idx in range(n_intervals):
   file.write("{:18.6f}".format(epoch_time[bin_idx]))
   file.write("{:12.3e}{:12.3e}{:12.3e}{:12.3e}\n".format(bfld_turb_eng_dens_avg[bin_idx], bmag_turb_eng_dens_avg[bin_idx],
                                                          alfv_turb_eng_dens_avg[bin_idx], vfld_turb_eng_dens_avg[bin_idx]))
file.close()
print("Superposed epoch analysis results for turbulent quantities saved to disk.")