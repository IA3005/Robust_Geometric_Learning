from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax

from pyriemann.utils.base import logm
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

from scipy.linalg import pinvh

from joblib import Parallel, delayed

import numpy as np  

#import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
#sys.path.append((os.path.dirname(__file__)))

from .tWishart import t_wish_est,log_generator_density,kurtosis_estimation, pop,pop_approx



def cart_prod(a,b):
    res=[]
    for i in range(a):
        for j in range(b):
            res.append([i,j])
    return res


def _check_metric(metric):
    if isinstance(metric, str):
        metric_mean = metric
        metric_dist = metric

    elif isinstance(metric, dict):
        # check keys
        for key in ['mean', 'distance']:
            if key not in metric.keys():
                raise KeyError('metric must contain "mean" and "distance"')

        metric_mean = metric['mean']
        metric_dist = metric['distance']

    else:
        raise TypeError('metric must be dict or str')

    return metric_mean, metric_dist


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metrics for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : list of ``n_classes`` ndarrays of shape (n_channels, \
            n_channels)
        Centroids for each class.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    """

    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) 
            or (n_matrices, n_freqs, n_channels, n_channels )
            Set of SPD matrices.
            
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.metric_mean, self.metric_dist = _check_metric(self.metric)
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if len(X.shape)==3:
            if self.n_jobs == 1:
                self.covmeans_ = [
                    mean_covariance(X[y == ll], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == ll])
                    for ll in self.classes_]
            else:
                self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(mean_covariance)(X[y == ll], metric=self.metric_mean,
                                             sample_weight=sample_weight[y == ll])
                    for ll in self.classes_)
        else: #len(X.shape)=4 
            if self.n_jobs == 1:
                self.covmeans_ = [[
                    mean_covariance(X[:,k][y == ll], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == ll])
                    for k in range(X.shape[1])]
                    for ll in self.classes_] 
                    
            else:

                self.covmeans_ = [Parallel(n_jobs=self.n_jobs)(
                    delayed(mean_covariance)(X[:,k][y == ll], metric=self.metric_mean,
                                             sample_weight=sample_weight[y == ll])
                    for k in range(X.shape[1]))
                    for ll in self.classes_]
            
        return self

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        n_centroids = len(self.covmeans_)
        if len(X.shape)==3:
            if self.n_jobs == 1:
                dist = [distance(X, self.covmeans_[m], self.metric_dist)
                        for m in range(n_centroids)]
            else:
                dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                    X, self.covmeans_[m], self.metric_dist)
                    for m in range(n_centroids)) #Nbclass-list of Ntestx1 tables
        else: #to improve: use np.stack
            if self.n_jobs == 1:
                dist_ = [[distance(X[:,k], self.covmeans_[m][k], self.metric_dist) 
                        for m in range(n_centroids)]
                         for k in range(X.shape[1])]
                #list of Ntestx1 arrays 
            else:
                dist_=[]
                for k in range(X.shape[1]):
                    dist_k= Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                            X[:,k], self.covmeans_[m][k], self.metric_dist)
                            for m in range(n_centroids))
                    dist_.append(dist_k)
                    
            dist = []
            for m in range(n_centroids):
                s =0
                for k in range(X.shape[1]):
                    s += dist_[k][m] 
                dist.append(s)

        dist = np.concatenate(dist, axis=1) #NtestxKxNbclass
        return dist

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            or (n_matrices, n_freqs, n_channels, n_channels )
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest centroid.
        """
        dist = self._predict_distances(X)
        #print(len(self.covmeans_),len(self.covmeans_[0]),self.covmeans_[0][0].shape)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            or (n_matrices, n_freqs, n_channels, n_channels )
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            or (n_matrices, n_freqs, n_channels, n_channels )
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)
    
    
class tWDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Bayesian classification by t-Wishart.
    """
    
    def __init__(self,n,dfs=[10],n_jobs=1,df_estimation_method=None,rmt=False,path=None):
        """Init.
        Parameters
        ----------
        n : int
            number of time samples.
        dfs : list, default=[10]
            degree(s) of freedom of the t- modeling (shape parameters) for different classes.
        n_jobs: int, default=1
            Number of jobs to run in parallel.
        df_estimation_method: str, default=None
            Method of estimating the degrees of freedom of the different classes
        rmt: bool, default=False
            if True, the RMT approximation is used.
        path: str, default=None
            Path to store the classifier attributes.
        """
        self.n = n 
        self.dfs = dfs
        self.n_jobs = n_jobs
        self.path = path
        self.df_estimation_method=df_estimation_method
        self.rmt = rmt
        if (self.dfs is None): #must be estimated then!
            if self.df_estimation_method is None:
                #if no estimation method is provided, then, use the simplest way 
                #which is the kurtosis estimation
                self.df_estimation_method = "kurtosis estimation"
            else:
                assert self.df_estimation_method in ["kurtosis estimation","pop exact","pop approx"],"Wrong estimation method for shape parameter"
        else:
            assert len(self.dfs)>0,"Empty list for `dfs` "
                      
    def estimate_df(self,S):
        if self.df_estimation_method=="kurtosis estimation":
            return kurtosis_estimation(S,self.n,np.mean(S,axis=0)/self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop exact":
            return pop(S,self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop approx":
            return pop_approx(S,self.n,rmt=self.rmt)
        
    def compute_class_center(self,S,df):
        if df==np.inf:
            return np.mean(S,axis=0)/self.n
        return t_wish_est(S,self.n,df=df)

    def fit(self, S, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : tWDA classifier instance
        """
        self.classes_ = np.unique(y)
        self.Nc = len(self.classes_)
        y = np.asarray(y)
        p,_ = S[0].shape

        #estimate dfs if needed        
        if self.dfs is None:
            if self.n_jobs==1:
                self.dfs = [self.estimate_df(S[y==self.classes_[i]]) for i in range(self.Nc)]
            else:
                self.dfs = Parallel(n_jobs=self.n_jobs)(delayed(self.estimate_df)(S[y==self.classes_[i]]) for i in range(self.Nc))
        else:
            if len(self.dfs)==1:
                self.dfs = [self.dfs[0] for _ in range(self.Nc) ]
        

        #estimate centers       
        if self.n_jobs==1:
            self.centers = [self.compute_class_center(S[y==self.classes_[i]],self.dfs[i]) for i in range(self.Nc)]
        else:
            self.centers = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(S[y==self.classes_[i]],self.dfs[i]) for i in range(self.Nc))
        
        #estimate proportions
        self.pi = np.ones(self.Nc)
        for k in range(self.Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
            
        #save centers and dfs
        if not(self.path is None):
            
            with open(self.path+"_centers.txt", "a") as fp:
                fp.write(str([self.centers[i].tolist() for i in range(self.Nc)]))
                fp.write("\n")
                
            with open(self.path+"_dfs.txt", "a") as fp:
                fp.write(str(self.dfs))
                fp.write("\n")
        return self
   
    def _predict_discimination(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        K,p,_ =covtest.shape
        discrimination = np.zeros((K,self.Nc)) #shape= (n_trials,n_classes)
        traces = np.zeros((K,self.Nc))
        
        if len(np.unique(np.asarray(self.dfs)))==1:
            #if a common df is used for all the classes:
            log_h = log_generator_density(self.n,p,self.dfs[0], neglect_df_terms=True)    
                
        for i in range(self.Nc):
            if len(np.unique(np.asarray(self.dfs)))!=1:
                #if different dfs are used for all classes
                log_h = log_generator_density(self.n,p,self.dfs[i])    

            center = self.centers[i].copy()
            inv_center = pinvh(center) #_matrix_operator(center,lambda x : 1/x)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                #discrimination between the center of the class i and the cov_j
                trace = np.matrix.trace(inv_center@covtest[j])
                traces[j,i] = trace
                discrimination[j,i] = np.log(self.pi[i])-0.5*self.n*logdet_center+log_h(trace)
                    
        return discrimination

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        covtest : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        preds : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        discrimination = self._predict_discimination(covtest)
        
        #save the discrimination table
        if not(self.path is None):
            with open(self.path+"_discrimination.txt", "a") as fp:
                fp.write(str(discrimination.tolist()))
                fp.write("\n")
                
        preds = []
        n_trials,n_classes = discrimination.shape
        for i in range(n_trials):
            preds.append(self.classes_[discrimination[i,:].argmax()])
        
        #save predictions
        if not(self.path is None):
            with open(self.path+"_preds.txt", "a") as fp:
                #np.save(fp,preds)
                fp.write(str(preds))
                fp.write("\n")
                
        preds = np.asarray(preds)
        return preds

    def transform(self, S):
        """get the distance to each centroid.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_discrimination(S)

    def fit_predict(self, S, y):
        """Fit and predict in one function."""
        self.fit(S, y)
        return self.predict(S)

    def predict_proba(self, S):
        """Predict proba using softmax.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(self._predict_discrimination(S))

   
class tWDA_noStack(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Bayesian classification by t-Wishart for `No stack` SSVEP paradigm
    """
    
    def __init__(self,n,dfs=[10],n_jobs=1,df_estimation_method=None,rmt=False,path=None):
        """Init.
        Parameters
        ----------
        n : int
            number of time samples.
        dfs : list or ndarray of shape (n_classes,n_freqs), default=[10]
            degree(s) of freedom of the t- modeling (shape parameters) for different classes and frequencies.
        n_jobs: int, default=1
            Number of jobs to run in parallel.
        df_estimation_method: str, default=None
            Method of estimating the degrees of freedom of the different classes and frequencies
        rmt: bool, default=False
            if True, the RMT approximation is used.
        path: str, default=None
            Path to store the classifier attributes
        """
        self.n = n 
        self.dfs = dfs
        self.n_jobs = n_jobs
        self.path = path
        self.df_estimation_method=df_estimation_method
        self.rmt = rmt
        if (self.dfs is None): #must be estimated then!
            if self.df_estimation_method is None:
                #if no estimation method is provided, then, use the simplest way 
                #which is the kurtosis estimation
                self.df_estimation_method = "kurtosis estimation"
            else:
                assert self.df_estimation_method in ["kurtosis estimation","pop exact","pop approx"],"Wrong estimation method for shape parameter"
        else:
            assert (type(self.dfs)==list and len(self.dfs)>0),"Empty list for `dfs` "
            
            
    def estimate_df(self,S):
        if self.df_estimation_method=="kurtosis estimation":
            return kurtosis_estimation(S,self.n,np.mean(S,axis=0)/self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop exact":
            return pop(S,self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop approx":
            return pop_approx(S,self.n,rmt=self.rmt)      
    
    def compute_class_center(self,S,df):
        if df==np.inf:
            return np.mean(S,axis=0)/self.n
        return t_wish_est(S,self.n,df=df)

    def fit(self, S, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_freqs, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : tWDA classifier instance
        """
        _,self.n_freqs,p,_ = S.shape
        self.classes_ = np.unique(y)
        self.Nc = len(self.classes_)
        y = np.asarray(y)
       
        #estimate dfs if needed        
        if self.dfs is None:
            #in this case, estimate dfs
            self.dfs = np.zeros((self.Nc,self.n_freqs))
            if self.n_jobs==1:
                for i in range(self.Nc):
                    for f in range(self.n_freqs):
                        self.dfs[i,f] = self.estimate_df(np.reshape(S[y==self.classes_[i],f,:,:],(len(y[y==self.classes_[i]]),p,p)))
                
            else:
                for f in range(self.n_freqs):
                    dfs_ = Parallel(n_jobs=self.n_jobs)(delayed(self.estimate_df)(np.reshape(S[y==self.classes_[i],f,:,:],(len(y[y==self.classes_[i]]),p,p))) for i in range(self.Nc))
                    for i in range(self.Nc):
                        self.dfs[i,f] = dfs_[i]       
        else:
            if len(self.dfs)==1:
                self.dfs = self.dfs[0]*np.ones((self.Nc,self.n_freqs))
            else:
                assert self.dfs.shape==(self.Nc,self.n_freqs),"The shape of the `dfs` matrix must be (n_classes,n_freqs)"
        
        
        #estimate centers
        self.centers = np.zeros((self.Nc,self.n_freqs,p,p))
        if self.n_jobs==1:
           for i in range(self.Nc):
               for f in range(self.n_freqs):
                self.centers[i,f] = self.compute_class_center(np.reshape(S[y==self.classes_[i],f,:,:],(len(y[y==self.classes_[i]]),p,p)),self.dfs[i+f*self.n_freqs])
        else:
            for f in range(self.n_freqs):
                centers_ = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(np.reshape(S[y==self.classes_[i],f,:,:],(len(y[y==self.classes_[i]]),p,p)),dfs_[i])  for i in range(self.Nc))
                for i in range(self.Nc):
                    self.centers[i,f] =centers_[i]
            
        #estimate proportions            
        self.pi = np.ones(self.Nc)
        for k in range(self.Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
            
        #save centers and dfs
        if not(self.path is None):
            with open(self.path+"_centers.txt", "a") as fp:
                fp.write(str([self.centers[i].tolist() for i in range(self.Nc)]))
                fp.write("\n")
                
            with open(self.path+"_dfs.txt", "a") as fp:
                fp.write(str(self.dfs))
                fp.write("\n")
        return self
    
    def _predict_discrimination(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        K,_,p,_ =covtest.shape
        discrimination = np.zeros((K,self.Nc))
        disc = np.zeros((K,self.Nc,self.n_freqs)) #shape= (n_trials,n_classes)
        
        if len(np.unique(np.asarray(self.dfs)))==1:
            #if a common df is used for all the classes:
            log_h = log_generator_density(self.n,p,self.dfs[0], neglect_df_terms=True)    
                
        
        for i in range(self.Nc):
            for f in range(self.n_freqs):
                if len(np.unique(np.asarray(self.dfs)))!=1:
                    log_h = log_generator_density(self.n,p,self.dfs[i,f])    

                center= self.centers[i,f].copy()
                inv_center = pinvh(center) #_matrix_operator(center,lambda x : 1/x)
                logdet_center = np.trace(logm(center))
                for j in range(K):
                    #distance between the center of the class i and the cov_j
                    trace = np.matrix.trace(inv_center@covtest[j,f,:,:])
                    disc[j,i,f] = -0.5*self.n*logdet_center+log_h(trace)
                    
        for j in range(K):
            for i in range(self.Nc):
                s = 0
                for f in range(self.n_freqs):
                    s += disc[j,i,f]
                discrimination[j,i] = s+ np.log(self.pi[i])
        return discrimination

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        covtest : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        preds : ndarray of int, shape (n_trials, 1)
            the predictions for all trials.
        """
        discrimination = self._predict_discrimination(covtest)
        
        #save discrimination
        if not(self.path is None):
            with open(self.path+"_discrimination.txt", "a") as fp:
                fp.write(str(discrimination.tolist()))
                fp.write("\n")
                
        preds = []
        n_trials,_ = discrimination.shape
        for i in range(n_trials):
            preds.append(self.classes_[discrimination[i,:].argmax()])
        
        #save predictions
        if not(self.path is None):
            with open(self.path+"_preds.txt", "a") as fp:
                fp.write(str(preds))
                fp.write("\n")
                
        preds = np.asarray(preds)
        return preds

    def transform(self, S):
        """get the distance to each centroid.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_discrimination(S)

    def fit_predict(self, S, y):
        """Fit and predict in one function."""
        self.fit(S, y)
        return self.predict(S)

    def predict_proba(self, S):
        """Predict proba using softmax.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(self._predict_discrimination(S))


    
