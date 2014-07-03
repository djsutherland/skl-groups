import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..features import Features

class BagMean(BaseEstimator, TransformerMixin):
    '''
    Transforms a list of features or :class:`Features` object by just taking
    the mean of each entry.

    Examples
    --------
    If you have a kernel where you can compute an (approximate?) explicit
    embedding, you can compute an explicit embedding of the mean map kernel
    (MMK, also known as maximum mean discrepancy or MMD) by first doing the
    kernel embedding and then taking the mean.
    Using :class:`sklearn.kernel_approximation.RBFSampler`::

        rbf_mmk = Pipeline([
            ('rbf', RBFSampler()),
            ('mean', BagMean()),
        ])
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        "Do nothing; this is a stateless transformer."
        return self

    def transform(self, X):
        '''
        Transform a list of bag features into its bag-of-words representation.

        Parameters
        ----------
        X : :class:`Features` or list of bag feature arrays
            Data to transform.

        Returns
        -------
        X_new : array, shape [len(X), X.dim]
            X transformed into its means.
        '''
        X = Features(X)
        return np.vstack([np.mean(bag, axis=0) for bag in X])
