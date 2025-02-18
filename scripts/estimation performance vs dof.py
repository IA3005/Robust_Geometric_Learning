import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from joblib import Parallel, delayed
from tqdm import tqdm
from src.manifold import generate_random_SPD
from src.tWishart import t_wishart_rvs,t_wish_est
import seaborn as sns
from scipy.linalg import pinv
from scipy.stats import ortho_group, norm, uniform,beta
from scipy.linalg import pinvh

#set parameters
n = 100
p= 10
cond = np.sqrt(10*p)
K = 1000
df0 = np.inf
MC= 50
seed=123
rmt = True
path = "results/perf_vs_dof/"
if not(os.path.exists(path)):
    os.makedirs(path)

exp=f"n={n}_p={p}_cond={int(cond)}_K={K}_df={df0}_MC={MC}_rmt={rmt}"
#list of dfs used for estimating the center ie to build the MLE
dfs = [1,3,5,10,50,100,200,500,1000,5e3,1e4,5e4,1e5,5e5,1e6] 
if df0 == 1e3:
    dfs = [1,5,10,50,100,500, 995,1000,1005, 5000,1e4,1e5,1e7,1e8]
if df0 == 1e4:
    dfs = [1,5,10,100,1000,5000,9950,9995,1e4,10005,10050,5e4,1e5,1e7,1e8]
if df0== 10: 
    dfs = [1,3,5,9,10,11,15,20,50,100,250,500,1000,5e3,1e4,5e4,1e5] 
if df0== 100: 
    dfs = [1,5,10,50,90,95,100,105,110,200,500,1000,5e3,1e4,5e4,1e5] 
if df0== 50: 
    dfs = [1,5,10,40,45,50,55,60,100,500, 1000,1e4,1e5,1e7,1e8] 
    
# Generate random center and samples from t-Wishart distribution
center = generate_random_SPD(p,cond,random_state=seed)
print("Generate samples:")
samples = []
for i in range(MC):
    samples.append(t_wishart_rvs(n,center,df0,size = K,random_state=seed+i))
print(len(samples),samples[0].shape)
print("  >Done!")
    
centers_estim = {df: None for df in dfs}
print("Compute the t-Wishart MLEs with RCG:")
for df in tqdm(dfs):
    centers_estim[df]=Parallel(n_jobs=-1)(delayed(t_wish_est)(samples[j],n,df) for j in (range(MC)))
print("  >Done!")
    
print("Compute the Wishart MLE:")
centers_wishart = [np.mean(samples[j],axis=0)/n for j in range(MC)]
print("  >Done!")

errors_estim = (np.asarray([[t_wish_dist(centers_estim[df][j],center,n,df0) for j in range(MC)] for df in dfs]))
errors_wishart = (np.asarray([t_wish_dist(centers_wishart[j],center,n,df0) for j in range(MC)]))

errors_estim_lowerQs,errors_estim_medians,errors_estim_upperQs = np.percentile(errors_estim,[5,50,95],axis=1)
errors_estim_means = np.mean(errors_estim,axis=1)
errors_estim_stds = np.std(errors_estim,axis=1)

errors_wishart_lowerQ,errors_wishart_median,errors_wishart_upperQ=np.percentile(errors_wishart,[5,50,95])
errors_wishart_mean = np.mean(errors_wishart)
errors_wishart_std = np.std(errors_wishart)

# plot figure
plt.fill_between(dfs,errors_estim_lowerQs,errors_estim_upperQs,color='royalblue',alpha=.5,linewidth=0)
plt.plot(dfs,errors_estim_medians,"royalblue",linewidth=2)
plt.plot(dfs,errors_estim_means,'r+')
plt.hlines(errors_wishart_median,min(dfs),max(dfs),'m',linewidth=1,label="wishart")
plt.hlines(errors_wishart_lowerQ,min(dfs),max(dfs),'m','dashed',linewidth=1)
plt.hlines(errors_wishart_upperQ,min(dfs),max(dfs),'m','dashed',linewidth=1)

# save results in textfile
filename= path+"perf_"+exp+".txt"
with open(filename,'w') as f:
    f.write("df,error_mean,error_std,error_median,error_qmin,error_qmax\n")
    for i in range(len(dfs)):
        f.write(f"{dfs[i]},{errors_estim_means[i]},{errors_estim_stds[i]},{errors_estim_medians[i]},{errors_estim_lowerQs[i]},{errors_estim_upperQs[i]}\n")
    #wishart
    f.write(f"{max(dfs)},{errors_wishart_mean},{errors_wishart_std},{errors_wishart_median},{errors_wishart_lowerQ},{errors_wishart_upperQ}\n")
