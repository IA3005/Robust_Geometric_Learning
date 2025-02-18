import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, ClusterMixin
from sklearn.utils.extmath import softmax
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance
from tqdm import tqdm
from scipy.linalg import pinvh
from pyriemann.utils.base import logm
from src.tWishart import t_wish_est,log_generator_density,kurtosis_estimation, pop,pop_approx



def kmeansplusplus_init(S,n_clusters,metric,n_jobs,random_state):
    """
    Kmeans++ initialization

    Parameters
    ----------
    S : ndarray, shape (K,p,p)
        SPD matrices to be clustered.
    n_clusters : int
        number of clusters.
    metric : str
        Metric for pairwise distance. For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`..
    n_jobs : int
        number of jobs to run in parallel.
    random_state : int, RandomState instance or None
        Controls the pseudo random number generation for choosing the next 
        centroid randomly with probability proportional to the squared distances

    Returns
    -------
    indexes : list
        list of the indexes of the chosen initial centroids.

    """
    rng = np.random.RandomState(random_state)
    indexes = []
    K,p,_ = S.shape
    remaining_idx = list(range(0,K))

    # Choose first centroid randomly
    idx = rng.randint(0, K)
    indexes.append(idx)
    remaining_idx.remove(idx)#possible candidates, all samples except the first chosen centroid
    

    #next centroids
    for l in tqdm(range(1, n_clusters)):
        # Compute distances of possible candidates to already-computed centroids
        if n_jobs == 1:
            distances =  [distance(S[indexes],S[j],metric) for j in remaining_idx]
        else:
            distances = Parallel(n_jobs=n_jobs)(
                    delayed(distance)(S[indexes],S[j],metric) for j in remaining_idx)
        
        distances = np.asarray(distances).reshape((K-l,l))
        
        #compute the squared distance of each sample to the closest centroid
        squared_distances = np.min(np.asarray(distances)**2,axis=1)
        assert len(squared_distances)==K-l,"Wrong Kmeans++ init!"
        
        # Choose next centroid randomly with probability proportional to the squared distances
        idx = rng.choice(remaining_idx, p=squared_distances/np.sum(squared_distances))
        indexes.append(idx)
        remaining_idx.remove(idx)
    return indexes


class tW_clustering(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    classification by t-Wishart distribution

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the MLE t-Wishart estimator. 
    Then, for each new point, the class is affected according to 
    the maximum discrimination.

    Parameters
    ----------
    dfs : list or None
        Degrees of freedom (shape parameters) for different clusters
        if None, they must be estimated.
    estimator : str, default="scm"
        Covariance matrix estimator.
    df_estimation_method : str or None, default="kurtosis_estimation"
        Estimation method for the degrees of freedom (if None)
    n_jobs : int, default=1
        Number of jobs.
    rmt : bool, default=False
        if True, the RMT approximation is used for the degrees
        of freedom estimation.
        
    Attributes
    ----------
    Nc : int
        Number of classes 
    n : int
        number of time samples
    classes_ : ndarray, shape (Nc,)
        Labels for each class.
    covmeans_ : list of ``Nc`` ndarrays of shape (n_channels, n_channels)
        Centroids for each class.
    dfs : list of ``Nc`` floats
        Degrees of freedom (shape parameters) for different classes
    pi : array, shape(Nc,)
        Proportions for each class.
    """

    def __init__(self,dfs, estimator = "scm",df_estimation_method="kurtosis_estimation",
                  n_jobs=1,rmt=False):
        """
        Init
        """
        self.dfs = dfs
        self.estimator = estimator
        self.df_estimation_method = df_estimation_method
        self.n_jobs = n_jobs
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

    def compute_class_center(self,S,df):
        """estimate the MLE t-Wishart as the class center
        """
        if df==np.inf:
            return np.mean(S,axis=0)/self.n
        return t_wish_est(S,self.n,df=df)

    def estimate_df(self,S):
        """estimate the degree of freedom of a given class
        """
        if self.df_estimation_method=="kurtosis estimation":
            return kurtosis_estimation(S,self.n,np.mean(S,axis=0)/self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop exact":
            return pop(S,self.n,rmt=self.rmt)
        if self.df_estimation_method=="pop approx":
            return pop_approx(S,self.n,rmt=self.rmt)
           
    def fit(self, X, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of trialss.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : tWDA classifier instance
        """
        self.classes_ = np.unique(y)
        self.n = X.shape[2] #n_times
        cov = Covariances(estimator=self.estimator).fit(X)
        S = cov.transform(X)
        p,_ = S[0].shape
        self.Nc = len(self.classes_)
        y = np.asarray(y)

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
            self.covmeans_ = [self.compute_class_center(S[y==self.classes_[i]],self.dfs[i]) for i in range(self.Nc)]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(S[y==self.classes_[i]],self.dfs[i]) for i in range(self.Nc))
        
        #estimate proportions
        self.pi = np.ones(self.Nc)
        for k in range(self.Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
            
        return self  

    def _predict_discimination(self, X):
        """Helper to predict the distance. equivalent to transform."""
        K,p,_ =X.shape
        cov = Covariances(estimator=self.estimator).fit(X)
        covtest = cov.transform(X)
        discrimination = np.zeros((K,self.Nc)) #shape= (n_trials,n_classes)
        traces = np.zeros((K,self.Nc))
        
        if len(np.unique(np.asarray(self.dfs)))==1:
            #if a common df is used for all the classes:
            log_h = log_generator_density(self.n,p,self.dfs[0], neglect_df_terms=True)    
                
        for i in range(self.Nc):
            if len(np.unique(np.asarray(self.dfs)))!=1:
                #if different dfs are used for all classes
                log_h = log_generator_density(self.n,p,self.dfs[i])    

            center = self.covmeans_[i].copy()
            inv_center = pinvh(center) #_matrix_operator(center,lambda x : 1/x)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                #discrimination between the center of the class i and the cov_j
                trace = np.matrix.trace(inv_center@covtest[j])
                traces[j,i] = trace
                discrimination[j,i] = np.log(self.pi[i])-0.5*self.n*logdet_center+log_h(trace)
                    
        return discrimination

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each trial according to the closest centroid.
        """
        discrimination = self._predict_discimination(X)
        preds = []
        n_trials,n_classes = discrimination.shape
        for i in range(n_trials):
            preds.append(self.classes_[discrimination[i,:].argmax()])
        
        return preds        

    def transform(self, X):
        """Get the discrimination to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.

        Returns
        -------
        discrimination : ndarray, shape (n_matrices, n_classes)
            Discrimination to each centroid.
        """
        return self._predict_discimination(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of discrimination.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(self._predict_discrimination(X))
    

class KMeans_tW(BaseEstimator, ClusterMixin):
    """t-Wishart based KMeans clustering.
    """

    def __init__(self, dfs=None,n_clusters=16, n_jobs=1, tol=1e-3,
                 df_estimation_method="kurtosis_estimation",init="kmeans++",
                 max_iter=100,rmt=False,estimator="scm",
                n_init=10, random_state: int = 42, verbose=0):
        self.n_clusters = n_clusters
        self.dfs =dfs
        self.df_estimation_method = df_estimation_method
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.estimator = estimator
        self.rmt = rmt
        self.tW = [tW_clustering(dfs=self.dfs,estimator=self.estimator,df_estimation_method=self.df_estimation_method,n_jobs=self.n_jobs,rmt=self.rmt) for _ in range(n_init)]
        self.verbose = verbose
        self.random_state = random_state
        self.n_init = n_init
        self.rng = np.random.RandomState(self.random_state)
        self.fitted = False
        self.init = init
        assert self.init in ["kmeans++","random"],"Wrong initilization method! Must be 'kmeans++' or 'random' "

    def _init_random_labels(self, X, seed):
        """Initialize the labels randomly."""
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]
        labels = rng.randint(self.n_clusters, size=n_samples)
        steps = 0
        while len(np.unique(labels)) < self.n_clusters:
            labels = self._init_random_labels(X, seed+self.n_init+steps)
            steps += 1
        return labels

    def fit(self, X, y=None):

        seeds = self.rng.randint(
            np.iinfo(np.int32).max, size=self.n_init)
        labels_list = []
        inertias_list = []
        self.initial_centroids_indexes = {}
        
        cov = Covariances(estimator=self.estimator).fit(X)
        S = cov.transform(X)
        
        for i, seed in enumerate(seeds):

            # Kmeans one init
            # -------------------------------------------------------------
            if self.verbose > 0:
                print(f'KMeans_tW: init {i+1}/{self.n_init}')

            # Initialize labels randomly
            if self.init=="kmeans++":
                initial_centroids_idx = kmeansplusplus_init(S, self.n_clusters, "riemann", self.n_jobs, seed)
                initial_centroids = S[initial_centroids_idx]
                if self.n_jobs==1:
                    initial_dists = [distance(initial_centroids,S[j],"riemann") for j in range(S.shape[0])]
                else:
                    initial_dists = Parallel(n_jobs=self.n_jobs)(
                        delayed(distance)(initial_centroids,S[j],"riemann") for j in range(S.shape[0]))
                initial_dists = np.asarray(initial_dists).reshape((S.shape[0],self.n_clusters))# shape=(K,n_clusters)
                labels = np.asarray([np.argmin(initial_dists[j,:]) for j in range(S.shape[0])])
                if self.verbose > 0:
                    #print(labels.shape)
                    #print(initial_dists.shape)
                    print("Kmeans++ init : Done! for seed",seed)
                    print("chosen centroids for init=",initial_centroids_idx)
                self.initial_centroids_indexes[seed]= initial_centroids_idx
            
            else:
                labels = self._init_random_labels(X, seed)
                
            # Compute initial centroids with RMT-MDM
            self.tW[i].fit(X, labels)

            # Iterate until convergence
            delta = np.inf
            n_iter = 0

            if self.verbose > 0:
                p_bar = tqdm(total=self.max_iter, desc='KMeans_tW', leave=True)

            while delta > self.tol and n_iter < self.max_iter:

                # Compute discrimination to centroids
                dist2 = self.tW[i].transform(X)

                # Assign each sample to the closest centroid
                labels_new = np.argmax(dist2, axis=1)

                # Compute new centroids
                self.tW[i].fit(X, labels_new)

                # Compute delta
                delta = np.sum(labels_new != labels)/len(labels)
                labels = labels_new
                n_iter += 1

                if self.verbose > 0:
                    p_bar.update(1)
                    p_bar.set_description(f'KMeans_tW (delta={delta})')

            # compute inertia
            dist2 = self.tW[i].transform(X)
            inertia = -np.sum([np.sum(dist2[labels==i, i])
                            for i in range(len(self.tW[i].covmeans_))])
            if self.verbose > 0:
                print(f'KMeans_tW: init {i+1}/{self.n_init} - inertia: {inertia:.2f}')
            labels_list.append(labels)
            inertias_list.append(inertia)
            if self.verbose > 0:
                print('-'*120+'\n')
            # ------------------------------------------------------------------------------------

        # Choose best init
        if self.verbose > 0:
            print('KMeans_tW: choosing best init...')
            print(f'inertias: {inertias_list}')
        best_init = np.argmin(inertias_list)
        labels = labels_list[best_init]
        inertia = inertias_list[best_init]
        self.inertia_ = inertia
        self.tW = self.tW[best_init]
        self.dfs = self.tW.dfs
        self.labels_ = labels
        self.fitted = True

        return self

    def predict(self, X):
        return self.tW.predict(X)

    def transform(self, X, y=None):
        if not self.fitted:
            self.fit(X)
        return self.labels_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
