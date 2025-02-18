import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys

from src.manifold import generate_random_SPD
from src.tWishart import t_wishart_rvs,t_wish_est,fixed_point,t_wish_dist





if __name__ == '__main__':
    n = 100 #degree of freedom (number of time samples)
    p = 10 #shape of samples (number of electrods)
    cond = 10 #conditionning number of the true center matrix
    df = 100 #degree of freedom (also called shape parameter) of the t- modeling
    K=300 #number of samples
    MC = 20 #number of Monte Carlo runs
    seed = 123 #random seed for the true center matrix
    thresholdFP = 5e-4 #threshold for the stopping criterion of the fixed-point algorithm
    path = "results/comapre FP and RCG results/" #path to store results
    save_results = False #if true, enables to store results in text files
    
    ## generate the true center matrix `Sigma`
    Sigma = generate_random_SPD(p,cond,random_state=seed)

    
    itersFP_ = []
    itersRCG_=[]
    
    timesFP_ = []
    timesRCG_=[]
    
    mseFP_=[]
    mseRCG_=[]
    
    for i in tqdm(range(MC)):
        
        #generate `K`` random matrices drawn from t- Wishart with `Sigma` as true center
        #and `df`as true shape parameter
        samples = t_wishart_rvs(n,Sigma,df,K,random_state=seed+i)
    
        #run estimation algorithms
        estimsFP,timesFP,errorsFP = fixed_point(samples,n,df,threshold=thresholdFP,maxiters=100000)
        estimsRCG,timesRCG,errorsRCG = t_wish_est(samples,n,df,algo="RCG",log_verbosity=2)
        
        #number of required iterations for each algorithm
        itersFP = len(estimsFP)
        itersRCG = len(estimsRCG)
        itersFP_.append(itersFP)
        itersRCG_.append(itersRCG)
        
        #required times for iterations for each algorithm 
        timesFP_.append([timesFP[j]-timesFP[0] for j in range(len(timesFP))])
        timesRCG_.append([timesRCG[j]-timesRCG[0] for j in range(len(timesRCG))])
       
        #MSEs (in dB) of all iterations for each algorithm
        mseFP = [20*np.log10(t_wish_dist(estimsFP[j],Sigma,n,df)) for j in range(itersFP)]
        mseRCG = [20*np.log10(t_wish_dist(estimsRCG[j],Sigma,n,df)) for j in range(itersRCG)]
        mseFP_.append(mseFP)
        mseRCG_.append(mseRCG)
        
        
    if save_results:
        if not(os.path.exists(path)):
            os.makedirs(path)
        filename = path+f"FP_n{n}_p{p}_df{df}_K{K}_MC{MC}_cond{cond}.txt"    
        with open(filename,'w') as f:
            f.write("MCiter,iter,err,time\n")
            for i in range(MC):
                for j in range(len(timesFP_[i])):
                    f.write(f"{i+1},{j+1},{mseFP_[i][j]},{timesFP_[i][j]}\n")
        
        filenamebis = path+f"RCG_n{n}_p{p}_df{df}_K{K}_MC{MC}_cond{cond}.txt"    
        with open(filenamebis,'w') as fbis:
            fbis.write("MCiter,iter,err,time\n")
            for i in range(MC):
                for j in range(len(timesRCG_[i])):
                    fbis.write(f"{i+1},{j+1},{mseRCG_[i][j]},{timesRCG_[i][j]}\n")
    
    
    ## plot mse vs iterations
    fig = plt.figure()
    for i in range(MC):
        if i==0:
            plt.plot(list(range(1,1+itersFP_[i])),mseFP_[i],c='r',label="FP",linewidth=1)
            plt.plot(list(range(1,1+itersRCG_[i])),mseRCG_[i],c='b',label="RCG",linewidth=1)
    
        else:
            plt.plot(list(range(1,1+itersFP_[i])),mseFP_[i],c='r',linewidth=1)
            plt.plot(list(range(1,1+itersRCG_[i])),mseRCG_[i],c='b',linewidth=1)
    
    plt.title(f"MSE along iterations \n n={n} p={p} df={df} K={K} MC={MC}")
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('MSE (dB)')
    plt.show()
        
    
    ## plot mse vs time
    fig = plt.figure()
    for i in range(MC):
        if i==0:
            plt.plot(timesFP_[i],mseFP_[i],c='r',label="FP",linewidth=1)
            plt.plot(timesRCG_[i],mseRCG_[i],c='b',label="RCG",linewidth=1)
        else:
            plt.plot(timesFP_[i],mseFP_[i],c='r',linewidth=1)
            plt.plot(timesRCG_[i],mseRCG_[i],c='b',linewidth=1)
            
    plt.title(f"MSE vs time \n n={n} p={p} df={df} K={K} MC={MC}")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('MSE (dB)')
    plt.show()



       
                    
