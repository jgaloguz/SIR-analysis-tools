# This code performs a superposed epoch analysis of turbulence geometry decomposition between slab and 2D modes in the interplanetary magnetic field for a list of events.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import warnings
import sys

# Flag to remove (or not) SB
remove_SB = True

# Ratio of 2D to slab amplitude
def Pyy_Pxx(q, theta, C2_Cs):
   costq = np.cos(theta)**(q-1.0)
   sintq = np.sin(theta)**(q-1.0)
   Pxx_slab = (q + 1.0) * costq
   Pxx_twoD = 2.0 * C2_Cs * sintq
   Pyy_slab = Pxx_slab
   Pyy_twoD = q * Pxx_twoD
   return (Pyy_slab + Pyy_twoD) / (Pxx_slab + Pxx_twoD)

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
missing_threshold = int(0.2 * bin_full_width) # threshold of number of missing datapoints above which interval is discarded
para_threshold = np.deg2rad(-0.1) # threshold to discard parallel BV alignment
perp_threshold = np.deg2rad(90.1) # threshold to discard perpendicular BV alignment

n_segs = 8*24+1 # number of (overlapping) segments
seg_pad = (n_pts % bin_disp) // 2 # number of datapoints to pad on first and last segment centers
seg_width = 2700 # width of segments
seg_half_width = seg_width // 2 # half of seg_width
n_intervals = (n_pts+seg_width) // bin_disp + 1 # number of (overlapping) subintervals
bin_pad = ((n_pts+seg_width) % bin_disp) // 2 # number of datapoints to pad on first and last subinterval centers
seg_centers = np.linspace(bin_half_width+seg_half_width, n_pts+bin_half_width+seg_half_width, num=n_segs)
P_xx_segs = [[] for _ in range(n_segs)]
P_yy_segs = [[] for _ in range(n_segs)]
P_zz_segs = [[] for _ in range(n_segs)]
q_segs = [0.0 for _ in range(n_segs)]
theta_BV_segs = [[] for _ in range(n_segs)]
all_theta_BV = []

# Import spectral lengths and average over segments
CIR_micro_data = np.loadtxt("output/SEA_micro_AC_no_fit_PS_broken_fit_without_SBs_" + str(year_start) + "-" + str(year_end) + ".lst")
inertial_range_slope_avg = -CIR_micro_data[:,5]
q_avg_half_width = 24 * seg_half_width // 1350
for seg in range(n_segs):
   left_idx = np.max([0,seg-q_avg_half_width])
   right_idx = np.min([seg+q_avg_half_width,n_segs-1])
   q_segs[seg] = np.mean(inertial_range_slope_avg[left_idx:right_idx+1])

epoch_idx = 0 # zero epoch index
# Iterate over SI list to find turbulent quantities, and output results to file
for FD_idx in range(n_FD):
   print("Analyzing event", FD_idx+1)
   while year_SW[epoch_idx] < FD_list[FD_idx]:
      epoch_idx = epoch_idx + 1
   Br_CIR = SW_data[0][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   Bt_CIR = SW_data[1][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   Bn_CIR = SW_data[2][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   Vr_CIR = SW_data[3][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   Vt_CIR = SW_data[4][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   Vn_CIR = SW_data[5][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
   B_missing_CIR = SW_data[8][epoch_idx-half_n_pts-bin_half_width-seg_half_width:epoch_idx+half_n_pts+bin_half_width+seg_half_width+1]
# iterate over subintervals
   for bin_idx in range(n_intervals):
      # print(bin_idx, n_intervals)
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
         Vr_interval = Vr_CIR[bin_left:bin_right+1]
         Vt_interval = Vt_CIR[bin_left:bin_right+1]
         Vn_interval = Vn_CIR[bin_left:bin_right+1]

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
         P_xx = np.var(Bx_faf)
         P_yy = np.var(By_faf)
         P_zz = np.var(Bz_faf)

# Filter by segments
         for seg in range(n_segs):
            if seg_centers[seg] - seg_half_width <= bin_center <= seg_centers[seg] + seg_half_width:
               P_xx_segs[seg].append(P_xx)
               P_yy_segs[seg].append(P_yy)
               P_zz_segs[seg].append(P_zz)
               theta_BV_segs[seg].append(theta_BV)
         all_theta_BV.append(theta_BV)

# Sort in angular bins
n_ang_bins = 9
ang_bin_width = 0.5 * np.pi / n_ang_bins
P_xx_segs_ang_bins = [[[] for __ in range(n_ang_bins)] for _ in range(n_segs)]
P_yy_segs_ang_bins = [[[] for __ in range(n_ang_bins)] for _ in range(n_segs)]
Prat_segs_ang_bins = [[[] for __ in range(n_ang_bins)] for _ in range(n_segs)]
theta_BV_ang_bins = np.linspace(0.5 * ang_bin_width, 0.5 * (np.pi - ang_bin_width), num=n_ang_bins)
for seg in range(n_segs):
   for ang in range(np.size(theta_BV_segs[seg])):
      ang_bin = int(theta_BV_segs[seg][ang] / ang_bin_width)
      Prat_segs_ang_bins[seg][ang_bin].append(P_yy_segs[seg][ang]/P_xx_segs[seg][ang])
# Average within each bin
Prat_segs_ang_bins_avg = np.zeros((n_segs,n_ang_bins))
Prat_segs_ang_bins_std = np.zeros((n_segs,n_ang_bins))
for seg in range(n_segs):
   for ang in range(n_ang_bins):
      Prat_segs_ang_bins_avg[seg][ang] = np.exp(np.mean(np.log(Prat_segs_ang_bins[seg][ang])))
      Prat_segs_ang_bins_std[seg][ang] = np.exp(np.std(np.log(Prat_segs_ang_bins[seg][ang])))

# Fit curve
C2_Cs_segs_avg = [0.0 for _ in range(n_segs)]
for seg in range(n_segs):
   guess_param = [1.0]
   Pyy_Pxx_seg = partial(Pyy_Pxx, q_segs[seg])
   opt_param, cov = curve_fit(Pyy_Pxx_seg, theta_BV_ang_bins[1:], Prat_segs_ang_bins_avg[seg,1:],
                              p0=guess_param, maxfev=10000, bounds=([0.01],[100.0]))
   C2_Cs_segs_avg[seg] = opt_param[0]

# Plots
fig = plt.figure(figsize=(10, 8), layout='tight')
ax1 = fig.add_subplot(111, projection='rectilinear')

ax1.hist(np.rad2deg(all_theta_BV), bins=9, range=(0.0,90.0), edgecolor='k')
# ax1.set_title('Slab fraction for {:d}-{:d}'.format(year_start,year_end), fontsize=32)
ax1.set_xlabel('$\\theta_{BV}$ ($^\\circ$)', fontsize=20)
ax1.set_ylabel('counts', fontsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

plt.show()
plt.close(fig)

fig = plt.figure(figsize=(12, 8), layout='tight')
ax1 = fig.add_subplot(111, projection='rectilinear')

epoch_time = np.linspace(-4.0,4.0, num=n_segs)
ax1.plot(epoch_time, 1.0 / (1.0 + np.array(C2_Cs_segs_avg)))
ax1.set_title('Slab fraction for {:d}-{:d}'.format(year_start,year_end), fontsize=32)
ax1.set_ylabel('Epoch (days)', fontsize=20)
ax1.set_ylabel('% slab', fontsize=20)
ax1.set_ylim(0.0, 1.0)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

plt.show()
plt.close(fig)

colors = ['tab:blue','tab:orange','tab:green','tab:red']
labels = ['slow','compressed slow','compressed fast','fast']
n_segs_plot = len(labels)
theta_array_fit = np.linspace(0.0, 0.5*np.pi, num=200)
fig = plt.figure(figsize=(12, 10), layout='tight')
ax1 = fig.add_subplot(111, projection='rectilinear')

ax1.axhline(1.0, color='k', linestyle='--')
P_yy_P_xx_seg_avg = [[] for _ in range(n_segs_plot)]
P_yy_P_xx_seg_std = [[] for _ in range(n_segs_plot)]
for idx in range(n_segs_plot):
   seg = int((idx + 0.5) * (n_segs / n_segs_plot))
   P_yy_P_xx_seg_avg[idx] = Prat_segs_ang_bins_avg[seg,:].copy()
   P_yy_P_xx_seg_std[idx] = Prat_segs_ang_bins_std[seg,:].copy()
   for ang in range(n_ang_bins):
      P_yy_P_xx_seg_std[idx][ang] = P_yy_P_xx_seg_std[idx][ang] / np.sqrt(np.size(Prat_segs_ang_bins[seg][ang]))
   ax1.errorbar(np.rad2deg(theta_BV_ang_bins), P_yy_P_xx_seg_avg[idx], yerr=P_yy_P_xx_seg_std[idx], fmt='o', c=colors[idx])
   ax1.plot(np.rad2deg(theta_array_fit), Pyy_Pxx(q_segs[seg], theta_array_fit, C2_Cs_segs_avg[seg]), color=colors[idx], label=labels[idx])
   ax1.axhline(q_segs[seg], color=colors[idx], linestyle='--')
ax1.set_title('Power Decomposition in Field Aligned Frame {:d}-{:d}'.format(year_start,year_end), fontsize=24)
ax1.set_xlabel('$\\theta_{\\mathrm{BV}}$ ($^\\circ$)', fontsize=20)
ax1.set_ylabel('$P_{y}/P_{x}$', fontsize=20)
ax1.set_xlim(0.0, 90.0)
ax1.set_ylim(0.5, 2.0)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.legend(fontsize=16, loc=4)

plt.show()
plt.close(fig)

# Print results to terminal
for idx in range(n_segs_plot):
   seg = int((idx + 0.5) * (n_segs / n_segs_plot))
   print(labels[idx], ":")
   print("\tq =", q_segs[seg])
   print("\tC_2/C_s =", C2_Cs_segs_avg[seg])
   print("\tr_s =", 1.0 / (1.0 + C2_Cs_segs_avg[seg]))

# Output results
if remove_SB:
   SB_label = "without_SBs_"
else:
   SB_label = "with_SBs_"
file = open("output/SEA_turb_decomp_alt_" + SB_label + str(year_start) + "-" + str(year_end) + ".lst", "w")
for seg in range(n_segs):
   file.write("{:18.6f}".format(epoch_time[seg]))
   file.write("{:12.3e}".format(q_segs[seg]))
   file.write("{:12.3e}".format(C2_Cs_segs_avg[seg]))
   file.write("\n")
file.close()

file = open("output/SEA_turb_decomp_alt_fits_" + SB_label + str(year_start) + "-" + str(year_end) + ".lst", "w")
for ang in range(n_ang_bins):
   file.write("{:18.6f}".format(theta_BV_ang_bins[ang]))
   for seg in range(n_segs):
      file.write("{:12.3e}".format(Prat_segs_ang_bins_avg[seg][ang]))
      file.write("{:12.3e}".format(Prat_segs_ang_bins_std[seg][ang]/np.sqrt(np.size(Prat_segs_ang_bins[seg][ang]))))
   file.write("\n")
file.close()
print("Alternate turbulence decomposition results saved to disk.")
