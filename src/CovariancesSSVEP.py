import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances as COV


class Covariances(BaseEstimator, TransformerMixin):
    """Enables to handle no-stacked SSVEP trials 
    """
    
    def __init__(self, estimator='scm'):
        self.estimator = estimator
        

    def fit(self, X, y):
        return self

    def transform(self, X):
        if len(X.shape)==3:#n_trials, n_freqs*n_channels, n_times 
            covmats = COV(estimator=self.estimator).transform(X)
           
        else:
            #n_trials, n_freqs, n_channels, n_times 
            covmats =[]
            for k in range(X.shape[1]):
                covmat_ = COV(estimator=self.estimator).transform(X[:,k])
                #shape of covmat = n_trials,n_channels,n_channels
                covmats.append(covmat_)
            covmats = np.stack(covmats,axis=1)
            #n_trials,n_freqs,n_channels,n_channels
        return covmats
