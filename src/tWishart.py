import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import numpy as np
import numpy.linalg as la
from scipy.special import gammaln,betaln
from scipy.linalg import pinvh
from scipy.stats import wishart,beta
import time


import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient,SteepestDescent
from pyriemann.utils.distance import distance

from src.manifold import SPD


### simulate t-Wishart samples

def t_wishart_rvs(n,scale,df,size=1,random_state=None):
    """
    Draw random samples from a t-Wishart distribution.

    Parameters
    ----------
    n : int
        Degrees of freedom, must be greater than or equal to dimension of the scale matrix.
    scale : array_like
        Symmetric positive definite scale matrix of the distribution.
    df : float
        Degrees of freedom of the t- modelling.
    size : int
        Number of samples to draw (defaut 1).

    Returns
    -------
    ndarray
        Random variates of shape (`size`,`dim`, `dim`), where
            `dim` is the dimension of the scale matrix..

    """
    p,_=scale.shape
    assert n>=p,"The degree of freedom `n` must be greater than or equal to dimension of the scale matrix."
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size,random_state=random_state)
    qs = beta.rvs(a=df/2,b=n*p/2,size=size,random_state=random_state)
    vec = df*(1/qs-1)/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T) 



### cost and grad for t- Wishart 

def t_wish_cost(R,S,n,df):
    """
    computes the cost function (negative log-likelihood of t-Wishart up to a multiplicative positive constant)

    Parameters
    ----------
    R : array
        Symmetric positive definite matrix, plays the role of the distribution's center.
    S : ndarray
        Samples, must be symmetric definite positive matrices of the same shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    float
        The negative log-likelihood of the samples at `R` (divided by n*number of samples).

    """
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(-(df+n*p)/2*np.log(1+a/df))/n/k


def t_wish_egrad(R,S,n,df):
    """
    Computes the Riemannian gradient of the cost (with respect to the Fisher Information Metric of t-Wishart)    

    Parameters
    ----------
    R : array
        Symmetric positive definite matrix, plays the role of the distribution's center.
    S : ndarray
        Samples, must be symmetric definite positive matrices of the same shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    TYPE
        Riemannian gradient of the cost of samples at `R`.

    """
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',(df+n*p)/(df+a),S)
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)


def t_wish_dist(S1,S2,n,df):
    """
    distance between S1 and S2, induced by the fim of t-wishart
    """
    if df==np.inf:
        return np.sqrt(n/2)*distance(S1,S2,metric="riemann")
    p = S1.shape[1]
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    return manifold.dist(S1,S2)

def log_generator_density(n,p,df,neglect_df_terms=False):
    if df==np.inf:
        log_h = lambda t:-0.5*t
    else:
        if neglect_df_terms:
            log_h = lambda t:-0.5*(df+n*p)*np.log(1+t/df)
        else:
            log_h = lambda t:-0.5*(df+n*p)*np.log(1+t/df)-0.5*n*p*np.log(0.5*df)-betaln(df/2,n*p/2)+gammaln(n*p/2)

    return log_h


## estimation algorithms to estimate MLE for the center of t- Wishart

def t_wish_est(S,n,df,algo="RCG",log_verbosity=0):
    """
    computes iteratively the MLE for the center of samples drawn from 
    t-Wishart with parameters n and df using the Riemannian Gradient Decent 
    or Riemannian Conjugate Gradient algorithm.

    Parameters
    ----------
    S : ndarry, shape (K,p,p)
        samples, symmetric definite matrices.
    n : int
        Degrees of freedom.
    df : int
        Degrees of freedom of the t- modelling.
    algo : str, default="RCG"
        Type of first-order Riemannian optimization algorithm, 
        either "RCG" for conjugate gradient or "RGD" for gradient descent.
    log_verbosity : int, default=0
        Level of information logged by the optimizer while it operates, 
        if 0, logs only the MLE. if 2, logs more information at each iteration. 
    Returns
    -------
    #if log_verbosity==0
    array, shape (p,p)
        MLE of the center parameter.
        
    #if log_verbosity==0
    estims : list of arrays of shape (p,p)
        List of the estimated center at each iteration of the algorithm.
    times : list of floats
        List of the ending time for each iteration of the algorithm.
    errors : list of floats
        List of the norm gradient of the difference between the current estimated center
        and the previous estimated center along iterations.
    """
    #p = S.shape[1]
    t = time.time()
    init = np.mean(S,axis=0)/n #np.eye(S.shape[-1])
    if df==np.inf:
        if log_verbosity==0:
            return init 
        else:
            t = time.time() - t
            return [init],[t],[0]
            
    p = S.shape[1]
    alpha_ = n/2*(df+n*p)/(df+n*p+2)
    beta_ = n/2*(alpha_-n/2)
    manifold = SPD(p,alpha_,beta_)
    
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    assert algo in ["RCG","RGD"],"Wrong Algorithm Name"
    if algo=="RCG": #conjugate gradient
        optimizer = ConjugateGradient(verbosity=0,log_verbosity=log_verbosity)
    else:
        optimizer = SteepestDescent(verbosity=0,log_verbosity=log_verbosity)  
    try:
        optim = optimizer.run(problem, initial_point=init)
    except:
        try: 
            optim = optimizer.run(problem, initial_point=np.eye(p))
        except: 
            return None
    
    if log_verbosity==0:
        #return only the MLE
        return optim.point
    else:
        #return the list of estimated center, required time and error (gradient norm) at each iteration 
        estims = optim.log['iterations']['point']
        times = optim.log['iterations']['time']
        errors = optim.log["stopping_criteria"]["min_gradient_norm"]
        return estims,times,errors

def fixed_point(S,n,df, init =None, maxiters=1000,threshold=1e-10):
    """
    computes iteratively the MLE for the center of samples drawn from 
    t-Wishart with parameters n and df using the fixed-point algorithm.

    Parameters
    ----------
    S : ndarry, shape (K,p,p)
        samples, symmetric definite matrices.
    n : int
        Degrees of freedom.
    df : int
        Degrees of freedom of the t- modelling.
    init : array, shape (p,p). Default=None
        The initial point for the algorithm.
    maxiters : int, default=1000
        Maximum number of iterations of the algorithm.
    threshold : float, default=1e-10
        Threshold on the L2 norm of the difference betwwen succesive points.
    Returns
    -------
    estims : list of arrays of shape (p,p)
        List of the estimated center at each iteration of the algorithm.
    times : list of floats
        List of the ending time for each iteration of the algorithm.
    errors : list of floats
        List of the L2 norm of the difference between the current estimated center
        and the previous estimated center along iterations.
    """
    K,p,_ = S.shape
    if init is None:
        old_estm = np.mean(S,axis=0)/n
    else: 
        old_estm = init
    i = 0
    error = np.inf
    u = lambda t: (df+n*p)/(df+t)
    estims=[old_estm]
    start_time = time.time()
    times=[start_time]
    errors=[error]
    #estims.append(old_estm)
    while (i<maxiters) and (error>threshold):
        inv_old_estm = pinvh(old_estm)
        u_traces = u(np.tensordot(inv_old_estm,S,axes=((0,1),(1,2))))
        new_estm = np.tensordot(u_traces,S,axes=1)/(n*K)
        error= np.linalg.norm(new_estm-old_estm)
        estims.append(new_estm)
        i+=1
        old_estm = new_estm.copy()
        end_time = time.time()
        times.append(end_time)
        errors.append(error)
    return estims,times,errors
        

## estimate dof for t-Wishart

def shrinkage(samples,n,maxiter=100,threshold=5e-2,rmt=False,verbosity=0):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if rmt:
        kappa = (kappa+1)/(1-p/(n*K))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    #df_old = n*p
    dfs = [df_old]
    t= 0
    error =np.inf
    #print("ok")
    while (t<maxiter) and (error>threshold):
        mle = t_wish_est(samples,n,df_old) #simplify
        assert mle.shape==(p,p),"wrong dim with mle"
        theta = np.trace(center_wishart)/np.trace(mle)
        if rmt:
            theta = (1-p/(n*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    if verbosity ==0:
        return dfs[-1]
    return dfs

def notpop(samples,n,maxiter=100,threshold=5e-2,rmt=False,verbosity=0):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if rmt:
        kappa = (kappa+1)/(1-p/(n*K))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    #df_old = n*p
    dfs = [df_old]
    t= 0
    error =np.inf
    #print("ok")
    while (t<maxiter) and (error>threshold):
        mle = t_wish_est(samples,n,df_old) #simplify
        #inverse_mle = pinvh(mle)
        inverse_mle = pinv(mle).copy()
        inverse_mle = (inverse_mle+inverse_mle.T)/2
        theta = np.einsum("kij,ji->",samples,inverse_mle)/(n*K*p)
        if rmt:
            theta = (1-p/(n*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    if verbosity ==0:
        return dfs[-1]
    return dfs


def pop(samples,n,maxiter=5,threshold=5e-2,rmt=False,verbosity=0):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if rmt:
        kappa = (kappa+1)/(1-p/(n*K))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    #df_old = n*p
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        inverses_cov =np.zeros((K,p,p))
        for i in range(K):
            index_i= list(range(0,i))+list(range(i+1,K))
            cov_i = t_wish_est(samples[index_i],n,df_old) #simplify
            #print(cov_i)
            #inverses_cov[i,:,:] = pinvh(cov_i)
            inverse_mle = pinv(cov_i).copy()
            inverses_cov[i,:,:] = (inverse_mle+inverse_mle.T)/2
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        if rmt:
            theta = (1-p/(n*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    if verbosity ==0:
        return dfs[-1]
    return dfs


def pop_approx(samples,n,maxiter=10,threshold=5e-2,rmt=False,verbosity=0):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        cov= t_wish_est(samples,n,df_old)
        if cov is None:
            return None
        inverses_cov =np.zeros((K,p,p))
        traces = np.einsum("kij,ji->k",samples,pinvh(cov))
        for i in range(K):
            cov_i = cov - (df_old+n*p)/(df_old+traces[i])*samples[i]/(n*K)#simplify
            inverses_cov[i,:,:] = pinvh(cov_i)
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        if rmt:
            theta = (1-p/(1*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    if verbosity ==0:
        return dfs[-1]
    return dfs
   
def kurtosis_estimation(samples,n,center=None, rmt=False):
    #traces of whitened samples
    K,p,_ = samples.shape
    if center is None:
        center = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center))
    #kappa = (E(Q²)/E(Q)²)*(np/(np+2))-1
    kappa = ((n*p)/(n*p+2))*np.mean(traces**2)/(np.mean(traces)**2)-1
    if rmt:
        kappa = (kappa+1)/(1-p/(n*K))-1 #kappa+1=1/theta
    if kappa>0:
        return 4+2/kappa
    else:
        if kappa==0:
            return np.inf
    return 2 #if kappa<0, df<4 for eg we can choose df=2
