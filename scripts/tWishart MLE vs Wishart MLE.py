import os, sys
#sys.path.append((os.path.dirname(__file__)))

import numpy as np
import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import wishart, multivariate_normal

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent, ConjugateGradient

from src.manifold import SPD
from src.tWishart import t_wish_est, t_wishart_rvs

from joblib import Parallel, delayed




def wishart_t_simulate_data(p,n,kmax,df,nIt,cond=10,seed=123):
    # manifold
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    # random Wishart parameter
    R = manifold.random_point(cond=cond,random_state=seed)
    # simulated data with maximum number of samples
    Ss = t_wishart_rvs(n=n,scale=R,df=df,size=(nIt,kmax),random_state=seed)
    return R, Ss, manifold

def wishart_t_perform_estim(R,Ss,n,ks,df,manifold):
    nIt,_,p,_ = Ss.shape
    nK = np.size(ks)
    # init error measures
    err_wish = np.zeros((nIt,nK))
    err_wish_t = np.zeros((nIt,nK))
    # Monte-Carlo estimation loop
    for kix in range(nK):
        k = ks[kix]
        print(f"k={k}")
        for it in tqdm(range(nIt)):
            # select data
            S = Ss[it][:k]
            # compute Wishart estimator
            wish_est = np.mean(S, axis=0)/n
            # compute Wishart t estimator
            wish_t_est = t_wish_est(S,n,df)
            # compute errors
            err_wish[it,kix] = manifold.dist(R,wish_est)**2
            err_wish_t[it,kix] = manifold.dist(R,wish_t_est)**2
    return err_wish, err_wish_t



def wishart_t_write_results(filename,ks,err_wish,err_wish_t,write_result=True):
    # compute mean errors 
    err_wish_mean = np.mean((err_wish),axis=0)
    err_wish_t_mean = np.mean((err_wish_t),axis=0)

    # compute  std errors 
    err_wish_std = np.std((err_wish),axis=0)
    err_wish_t_std = np.std((err_wish_t),axis=0)
    
   
    # plot
    fig, ax = plt.subplots()

    ax.fill_between(ks,err_wish_mean-err_wish_std,err_wish_mean+err_wish_std,alpha=.5,linewidth=0)
    wish_line, = ax.plot(ks,err_wish_mean,linewidth=2)
    wish_line.set_label("Wishart estimator")


    ax.fill_between(ks,err_wish_t_mean-err_wish_t_std,err_wish_t_mean+err_wish_t_std,alpha=.5,linewidth=0)
    wish_t_line, = ax.plot(ks,err_wish_t_mean,linewidth=2)
    wish_t_line.set_label(f"MLE df={df}")
    
    
    crbs =  (0.5*p*(p+1)/np.asarray(ks))
    crb_line, = ax.plot(ks,crbs,linewidth=1,linestyle="dashed")
    crb_line.set_label("CRB")
    
    
    ax.legend()

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.show()
    
    # write in file
    if write_result:
        with open(filename,'w') as f:
            f.write("k,wishart_err_mean,wishart_err_std,twishart_err_mean,twishart_err_std,crb\n")
            for kix in range(nK):
                f.write(f"{ks[kix]},{err_wish_mean[kix]},{err_wish_std[kix]},{err_wish_t_mean[kix]},{err_wish_t_std[kix]},{crbs[kix]}\n")
    

if __name__ == '__main__':
    
    # simulation parameters
    p = 10
    n = 500
    cond = 10
    ks = [30,70,100,300,500,700,1000]
    kmax = np.max(ks)
    nK = np.size(ks)
    seed = 123
    # number of Monte-Carlo
    nIt = 200
    # Student degree of freedom
    df = 10
    path = "results/compare_estimators/"
    if not(os.path.exists(path)):
        os.makedirs(path)
    filename = path+f"estimation_n{n}_p{p}_df{df}_MC{nIt}_cond{cond}.txt"
    # simulate data and get manifold    
    R, Ss, manifold = wishart_t_simulate_data(p,n,kmax,df,nIt,cond=cond,seed=seed)
    
    # perform estimation
    err_wish, err_wish_t = wishart_t_perform_estim(R,Ss,n,ks,df,manifold)
    
    # analyze results    
    wishart_t_write_results(filename,ks,err_wish,err_wish_t)