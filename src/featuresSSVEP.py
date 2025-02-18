import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin





class ExtendedSSVEPSignal(BaseEstimator, TransformerMixin):
    """Prepare FilterBank SSVEP EEG signal for estimating extended 
    and no-extended covariances

    FilterBank SSVEP EEG are of shape (n_trials, n_channels, n_times, n_freqs)
    and should be converted in (n_trials, n_channels*n_freqs, n_times) if stack=True,
    otherwise, converted in (n_trials, n_freqs, n_channels, n_times)
    """

    def __init__(self,stack=True):
        self.stack =stack
        

    def fit(self, X, y):
        """No need to fit for ExtendedSSVEPSignal"""
        return self

    def transform(self, X):
        """Transpose and reshape EEG for extended covmat estimation"""
        out = X.transpose((0, 3, 1, 2))
        n_trials, n_freqs, n_channels, n_times = out.shape
        if self.stack:
            out = out.reshape((n_trials, n_channels * n_freqs, n_times))
        return out
