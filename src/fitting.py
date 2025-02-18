import numpy as np
from src.tWishart_ import t_wish_est, t_wishart_rvs, pop, notpop, shrinkage, kurtosis_estimation
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import wishart,kstest


def cdf(samples,x):
    """
    computes the empirical cumulative density function of samples at the quantile x i.e. proportion of samples that are <= x.
    
    Parameters
    ----------
    samples : array or list of floats.
        random floats
    x : float
        quantile

    Returns
    -------
    float
        the empirical cdf of samples at x.
    """
    card=0
    for j in range(len(samples)):
        if samples[j]<=x:
            card+=1
    return card/len(samples)

def estimate_df(S,n,df_estimation_method="pop exact",rmt=True):
    if df_estimation_method=="kurtosis estimation":
        return kurtosis_estimation(S,n,np.mean(S,axis=0)/n,rmt=rmt)
    if df_estimation_method=="pop exact":
        return pop(S,n,rmt=rmt)
    # if df_estimation_method=="pop approx":
    #     return pop_approx(S,n,rmt=rmt)
    if df_estimation_method=="notpop":
        return notpop(S,n,rmt=rmt)
    if df_estimation_method=="shrinkage":
        return shrinkage(S,n,rmt=rmt)

def fold_covs_per_class(cov_data):
    all_covs = []
    all_labels = []
    selected_subjs = list(cov_data.keys())
    for subject in selected_subjs:
        for i in range(len(cov_data[subject][0])):
            all_covs.append(cov_data[subject][0][i,:,:])
        all_labels.extend(cov_data[subject][1])
    all_covs = np.asarray(all_covs)

    unique_labels = np.unique(all_labels)
    inx_per_class={k:[] for k in unique_labels}
    for j in range(len(all_labels)):
        inx_per_class[all_labels[j]].append(j)
    return all_covs, all_labels, inx_per_class


def fit_eeg(quantity_to_fit,all_covs, all_labels, inx_per_class,n,
            df={'13':20, '17':20, '21':20, 'rest':20},
            iters=500,n_samples=100000,n_jobs=-1):
    """
    performs the fitting of real EEG data with the Wishart and the t-Wishart distributions.

    Parameters
    ----------
    quantity_to_fit : str
        Quantity on which we perform fitting. For example, if `trace`, the fitting of the trace of samples is considered.
    cov_data : dict
        Dictionary of covariance matrice and labels of each subject from the SSVEP-EEG dataset
    n : int
        Number of time samples
    df : dict, optional
        Degrees of freedom for each class (default {1:20, 2:20, 3:20, 4:20})
    iters : int, optional 
        Number of quantiles per class (defaut 500)
    n_samples : int,optional
        Number of Wishart/t-Wishart samples drawn to fit real data (default 100000)
    n_jobs : int,optional
        Number of jobs (default -1).

    Returns
    -------
    xs : dict
        Quantiles for each class.
    eeg_cdf : dict
        Empirical cumulative density function of EEG samples for each class 
        (for example, if quantity_to_fit is `norm`, the samples are the norms of the extracted covariance matrices)  
    wishart_cdf : dict
        Cumulative density function of Wishart samples for each class (the center of each class is estimated using the Wishart estimator)
    t_wishart_cdf : dict
        Cumulative density function of t-Wishart samples for each class (the center of each class is estimated using the MLE)
    tests_W : dict
        Results of Kolmogorov-Smirnov fitting test (statistic and p-value) with the Wishart distribution for each class
    tests_tW : dict
        Results of Kolmogorov-Smirnov fitting test (statistic and p-value) with the t-Wishart distribution for each class

    """
    parallel = Parallel(n_jobs=n_jobs, verbose=0)
    
    transform = None
    if quantity_to_fit=="trace":
        transform = lambda S : S.T.trace()
    if quantity_to_fit=="norm":
        transform = lambda S : np.linalg.norm(S,axis=(1,2)) 
    if quantity_to_fit=="det":
        transform = lambda S : -np.log10(np.linalg.det(S))
    if "tracepower" in quantity_to_fit: # E(tr(S)^k)
        power_ = float(quantity_to_fit.split('_')[-1])
        transform = lambda S : S.T.trace()**power_
    if "powertrace" in quantity_to_fit: # E(tr(S^k))
        power_ = int(quantity_to_fit.split('_')[-1])
        transform = lambda S : (S**power_).T.trace()
    if "normpower" in quantity_to_fit: # E(||S||^k)
        power_ = float(quantity_to_fit.split('_')[-1])
        transform = lambda S : np.linalg.norm(S,axis=(1,2))**power_
    if "powernorm" in quantity_to_fit: # E(||S^k||)
        power_ = int(quantity_to_fit.split('_')[-1])
        transform = lambda S : np.linalg.norm(S**power_,axis=(1,2))
    if quantity_to_fit=="largest_eigenvalue":
        transform = lambda S : np.linalg.eigvalsh(S)[:,-1]
    if quantity_to_fit=="smallest_eigenvalue":
        transform = lambda S : np.linalg.eigvalsh(S)[:,0]
         
    assert not(transform is None),"Wrong quantity to fit!"

    #1. compute the empirical cdf using the eeg samples 
            
    xs={}
    eeg_cdf={}
    eeg_samples={}
    
    unique_labels = np.unique(all_labels)
    for k in unique_labels: 
        eeg_samples[k] = transform(all_covs[inx_per_class[k]])
        xmin = np.min(eeg_samples[k])*0.9
        xmax = np.max(eeg_samples[k])*1.1
        xs[k] = np.linspace(xmin,xmax,iters)
        eeg_cdf[k] = parallel(delayed(cdf)(eeg_samples[k],x) for x in xs[k])


    #2. compute the center of each class with two estimators: 
        ## the Wishart estimator and the MLE of t-wishart 
    
    centers_t_wishart={}
    centers_wishart={}
    for k in unique_labels:
        centers_wishart[k] = np.mean(all_covs[inx_per_class[k]],axis=0)/n 
        centers_t_wishart[k] = t_wish_est(all_covs[inx_per_class[k]],n,df=df[k])
        

    #3. compute the empirical cdf using iid samples ~ Wishart whose center
        ## is the Wishart estimator for each class
        
    wishart_samples={}
    wishart_cdf = {}
    tests_W={} #results of Kolmogorov-Smirnov tests of wishart fitting
    
    for k in unique_labels:
        wishart_samples[k] = transform(wishart.rvs(n,centers_wishart[k],n_samples))
        wishart_cdf[k] = parallel(delayed(cdf)(wishart_samples[k],x) for x in xs[k])
        tests_W[k] = kstest(wishart_samples[k],eeg_samples[k])
        
        
    #4. compute the empirical cdf using iid samples ~ t-Wishart whose center
        ## is the t-Wishart MLE and with the given dof for each class
        
    t_wishart_samples={}
    t_wishart_cdf={}
    tests_tW={} #results of Kolmogorov-Smirnov tests of t-wishart fitting
    
    for k in unique_labels:
        t_wishart_samples[k] = transform(t_wishart_rvs(n,centers_t_wishart[k],df[k],n_samples))
        t_wishart_cdf[k] = parallel(delayed(cdf)(t_wishart_samples[k],x) for x in xs[k])
        tests_tW[k] = kstest(t_wishart_samples[k],eeg_samples[k])


    return xs,eeg_cdf,wishart_cdf,t_wishart_cdf,tests_W , tests_tW
    

    
