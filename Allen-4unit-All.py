import h5py
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import math
from scipy.optimize import nnls  

########################
# Part 1. Preprocessing

# Specify the path to your text file
file_path = './cell_id.txt'

# Read data from the text file
with open(file_path, 'r') as file:
    data_string = file.read()

# Extract session indices and arrays from the string
sessions = {}
for line in data_string.strip().split('\n'):
    parts = line.split(':')
    session_index = int(parts[0].split('=')[1].strip())
    array_str = parts[1].replace('[', '').replace(']', '').replace(',', '').split()
    array = [int(val) for val in array_str]
    sessions[session_index] = np.array(array)
    
cell_types = sessions 
print(cell_types[1])

# Check dataset 
with h5py.File('aggFC.mat', 'r') as file : 
    print(file.keys())
    print(file['sessions'])
    print(file['sesfcid'])
    print(file['sesfc'])
    print(file['sesfcPV'])
    
e_raw = []
p_raw = []
s_raw = []
v_raw = []

with h5py.File('aggFC.mat', 'r') as file:
    
    # Trial-averaged responses during visual presentation
    dataset = file['meanRttdgc_fcagg']
    dataset = dataset[0]
    
    # Loop through sessions(mice)
    for session_index, reference in enumerate(dataset):
        
        # Ignore 17-1, 18-1 th sessions
        if session_index in [16, 17]:
            continue
        
        response = file[reference]
        response = np.asarray(response)
        
        cell_type = cell_types[session_index]
        
        e_indices = np.where(cell_type == 1)[0]
        e_response = response[:, e_indices]
        e_raw.append(e_response)
        
        p_indices = np.where(cell_type == 2)[0]
        p_response = response[:, p_indices]
        p_raw.append(p_response)
        
        s_indices = np.where(cell_type == 3)[0]
        s_response = response[:, s_indices]
        s_raw.append(s_response)
        
        v_indices = np.where(cell_type == 4)[0]
        v_response = response[:, v_indices]
        v_raw.append(v_response)

        # Print information for each session in a single line
        print(f'Session {session_index + 1:2d}: Num total cells={response.shape[1]:3d}, Num e={e_response.shape[1]:3d}, Num p={p_response.shape[1]:3d}, Num s={s_response.shape[1]:3d}, Num v={v_response.shape[1]:3d}')
        
e_cat = np.concatenate(e_raw, axis=1)
p_cat = np.concatenate(p_raw, axis=1)
s_cat = np.concatenate(s_raw, axis=1)
v_cat = np.concatenate(v_raw, axis=1)

# Assuming e_cat, p_cat, s_cat, and v_cat are the concatenated arrays for 'e', 'p', 's', and 'v' cells
s_cat = np.concatenate(s_raw, axis=1)
v_cat = np.concatenate(v_raw, axis=1)

# Calculate mean and standard deviation along axis 1 (columns) for each neuronal type
s_mean = s_cat.mean(axis=1) 
s_std = s_cat.std(axis=1)/ np.sqrt(s_cat.shape[1])

v_mean = v_cat.mean(axis=1)
v_std = v_cat.std(axis=1)/ np.sqrt(v_cat.shape[1])

# Perform curve fitting
from scipy.optimize import curve_fit
def func(x, a):
    return a / x
x_obs = s_mean 
y_obs = v_mean 
popt, pcov = curve_fit(func, x_obs, y_obs)

# Extract the fitted parameter
a_fit = popt[0]

# Generate y values using the fitted curve
xx = np.linspace(np.min(x_obs), np.max(x_obs), 100)
yy_fit = func(xx, a_fit)

plt.plot(xx, yy_fit, 'k--', label='Fitted curve: y = {:.2f}/x'.format(a_fit))
# plt.scatter(s_mean, v_mean)
plt.errorbar(s_mean, v_mean, xerr=s_std, yerr=v_std, fmt='o')

plt.title('Each stimulus (4*9 data points)')
plt.xlabel('mean S rate')
plt.ylabel('mean V rate')
plt.legend()
plt.show() 

# Contrast data 
c = np.asarray([.01, .02, .04, .08, .13, .2, .35, .6, 1])

# Assuming e_cat, p_cat, s_cat, and v_cat are the concatenated arrays for 'e', 'p', 's', and 'v' cells
e_cat = np.concatenate(e_raw, axis=1)
p_cat = np.concatenate(p_raw, axis=1)
s_cat = np.concatenate(s_raw, axis=1)
v_cat = np.concatenate(v_raw, axis=1)

def reshape_cat_data (alpha_cat) :
    # Reshapes an array with shape (36,*) into (9,4*),
    # where * can be any positive integer.
    
    alpha_cat_reshape = np.array([alpha_cat[0:9].T, alpha_cat[9:18].T, alpha_cat[18:27].T, alpha_cat[27:].T]).T 
    alpha_cat_reshape = alpha_cat_reshape.reshape((9,-1))
    return alpha_cat_reshape

# Reshape concatenated arrays to ignore orientational information 
e_cat = reshape_cat_data(e_cat)
p_cat = reshape_cat_data(p_cat)
s_cat = reshape_cat_data(s_cat)
v_cat = reshape_cat_data(v_cat)

# Proportion of each neuronal type 
N = np.asarray([e_cat.shape[-1], p_cat.shape[-1], s_cat.shape[-1], v_cat.shape[-1]])


# Calculate mean and standard deviation along axis 1 (columns) for each neuronal type
e_mean = e_cat.mean(axis=1)
e_std = e_cat.std(axis=1) / np.sqrt(e_cat.shape[1])

p_mean = p_cat.mean(axis=1)
p_std = p_cat.std(axis=1)/ np.sqrt(p_cat.shape[1])

s_mean = s_cat.mean(axis=1)
s_std = s_cat.std(axis=1)/ np.sqrt(s_cat.shape[1])

v_mean = v_cat.mean(axis=1)
v_std = v_cat.std(axis=1)/ np.sqrt(v_cat.shape[1])


# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

# Plot 'e' cells
axs[0].errorbar(c, e_mean, yerr=e_std, label='e cells', color='gray', fmt= '-o', capsize=5)
axs[0].set_ylabel(r'Mean Rate of E')

# Plot 'p' cells
axs[1].errorbar(c, p_mean, yerr=p_std, label='p cells', color='lightseagreen', fmt= '-o', capsize=5)
axs[1].set_ylabel(r'Mean Rate of P')

# Plot 's' cells
axs[2].errorbar(c, s_mean, yerr=s_std, label='s cells', color='orange', fmt= '-o', capsize=5)
axs[2].set_ylabel(r'Mean Rate of S')
plt.legend()

# Plot 'v' cells
axs[3].errorbar(c, v_mean, yerr=v_std, color='hotpink', fmt= '-o', capsize=5, alpha=0.5)
# for i in range(8) : 
#     axs[3].plot(c, v_cat[:,i],  color='hotpink', linestyle='-', marker=None, alpha=0.5)

axs[3].set_ylabel(r'Mean Rate of V')

a_fit = 45.91
a_fit_uncertainty = 5.61
y_err_1 = s_std*a_fit/(s_mean**2)
y_err_2 = a_fit_uncertainty / s_mean 
y_err_tot = np.sqrt(y_err_1**2 + y_err_2**2)

v_mean_fit = a_fit / s_mean 
v_std_fit = y_err_tot

axs[3].errorbar(c, v_mean_fit, yerr=v_std_fit, color='black', fmt= '-o', capsize=5)
axs[3].set_ylabel(r'Mean Rate of V')

# Customize the plot as needed
axs[3].set_xlabel('Stimulus Contrast')
plt.tight_layout()
plt.legend()
plt.show()

q_mossing = [0.8, 0.08, 0.06, 0.06] # corrected
e, p, s, v = q_mossing[0]*e_mean, q_mossing[1]*p_mean, q_mossing[2]*s_mean, q_mossing[3]*v_mean_fit
e_std, p_std, s_std, v_std = q_mossing[0]*e_std, q_mossing[1]*p_std, q_mossing[2]*s_std, q_mossing[3]*v_std_fit
c = np.asarray([.01, .02, .04, .08, .13, .2, .35, .6, 1])
A = np.asarray([e, -p, -s, -v, np.ones(len(c)), c]).T
print (A) 

########################
# PART 2. Fitting 

def get_stable_solutions (c_idx, w, visualize=False) : 
    # Input : Contrast Index, Weight
    # c_idx : index of the contrast
    # w : parameter from NNLS
    
    # Output : Rate Evolved 
    
    # Parameters for rate evolution
    m = 500 # Total steps for evolution
    tau = 10 # (ms). Time constant of the rate model 
    Delta_t = 1 # (ms). Time interval for numerical integration 
    traj = [] # Rate trajectory stored every step. 
    eps = Delta_t / tau # = 0.2 

    # Transfer function of the rate model 
    def f(x):
        relu_result = np.maximum(0, x)
        square_result = relu_result ** 2
        return square_result

    # 1. Set initial configuration as data points
    r = np.asarray([e[c_idx], p[c_idx], s[c_idx], v[c_idx]])

    # 2. Integrate dynamicss
    for t in range(m) : 
        traj.append(r)
        # print(w.shape) # (4,6)
        # print(np.asarray([r[0], -r[1], -r[2], -r[3], 1, c[c_idx]]).shape) # (6,)
        r = r + eps * ( - r + f (np.dot (w,  np.asarray([r[0], -r[1], -r[2], -r[3], 1, c[c_idx]]) ) ) )
    traj = np.asarray(traj)
    
    # 3. Check Convergence 
    if visualize == True : 
        for i in range(4) : 
            plt.plot(traj[:,i])
        plt.show()
    
    return traj[-1,:]

# Output 
w_list = [] 
r_conv_list = [] 
llh_list = [] 

n_sample = 500

sampling_factor = 2

for k in tqdm(range(n_sample)) : 
    
    # 1. Sample Pseudo Data 
    re_sam = e + np.random.normal(0, sampling_factor *e_std)
    rp_sam = p + np.random.normal(0, sampling_factor *p_std)
    rs_sam = s + np.random.normal(0, sampling_factor *s_std)
    rv_sam = v + np.random.normal(0, sampling_factor *v_std)
    
    # 2. Fit NNLS Weight to the Sampled Data
    A = np.asarray([re_sam, -rp_sam, -rs_sam, -rv_sam, np.ones(len(c)), c]).T
    ye = np.sqrt(e); xe, ee = nnls(A,ye)
    yp = np.sqrt(p); xp, ep = nnls(A,yp)
    ys = np.sqrt(s); xs, es = nnls(A,ys)
    yv = np.sqrt(v); xv, ev = nnls(A,yv)
    w = np.asarray([xe, xp, xs, xv])
    
    # 3. Get Stable Solutions for NNLS Weight
    conv_r = [] 
    for c_idx in range(len(c)) : 
        conv_r.append( get_stable_solutions(c_idx,w,visualize=False) )
    conv_r = np.asarray(conv_r)
    re_conv = conv_r[:,0]
    rp_conv = conv_r[:,1]
    rs_conv = conv_r[:,2]
    rv_conv = conv_r[:,3]
    
    # plt.plot(c, conv_r[:,0], marker='o')
    
    # 4. Calculate Log-likelihood of The Solution & Data    
    llh = -0.5 * ( (re_conv - e)/(e_std) )**2 -0.5* np.log(2 * np.pi * e_std**2)
    llh += -0.5 * ( (rp_conv - p)/(p_std) )**2 -0.5* np.log(2 * np.pi * p_std**2)
    llh += -0.5 * ( (rs_conv - s)/(s_std) )**2 -0.5* np.log(2 * np.pi * s_std**2)
    llh += -0.5 * ( (rv_conv - v)/(v_std) )**2 -0.5* np.log(2 * np.pi * v_std**2)
    llh = llh.sum() 
    
    if not math.isnan(llh) :
        w_list.append(w)
        r_conv_list.append(conv_r)
        llh_list.append(llh)
        
# 5. Select Top Models 
w_list = np.asarray(w_list)
r_conv_list = np.asarray(r_conv_list)
llh_list = np.asarray(llh_list)
    
sorted_indices = sorted(range(len(llh_list)), key=lambda k: -llh_list[k])
print(len(llh_list))
plt.plot(llh_list[sorted_indices][:100]); plt.title(r'log likelihood(4-unit model; data)'); plt.xlabel('Model Indices')

########################
# Part 3. Analysis 

top_n = 50

for k in sorted_indices[:top_n] : 
    plt.plot(c, r_conv_list[k][:,0], color='gray',alpha=0.1)
    plt.plot(c, r_conv_list[k][:,1], color='lightseagreen',alpha=0.1)
    plt.plot(c, r_conv_list[k][:,2], color='orange',alpha=0.1)
    plt.plot(c, r_conv_list[k][:,3], color='hotpink',alpha=0.1)
    
plt.errorbar(x=c,y=e,yerr=e_std,color='gray',fmt= '-o', capsize=5)
plt.errorbar(x=c,y=p,yerr=p_std,color='lightseagreen',fmt= '-o', capsize=5)
plt.errorbar(x=c,y=s,yerr=s_std,color='orange',fmt= '-o', capsize=5)
plt.errorbar(x=c,y=v,yerr=v_std,color='hotpink',fmt= '-o', capsize=5)

plt.title('4-Unit Model (=1st Order Mean Field)')
plt.ylabel('Mean Firing Rate'); plt.xlabel('Contrast')
# plt.ylim(0,1)
plt.show()

w_LD = w_list[sorted_indices[:top_n]].mean(axis=0)[:,:4]
plt.matshow(np.log(w_LD), cmap='gray_r') # higher - darker
plt.colorbar(); plt.show()