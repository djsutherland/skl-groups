import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..features import as_features

class BagMean(BaseEstimator, TransformerMixin):
    '''
    Transforms a list of features or :class:`Features` object by just taking
    the mean of each entry.

    Examples
    --------
    If you have a kernel where you can compute an (approximate?) explicit
    embedding, you can compute an explicit embedding of
    :class:`skl_groups.kernels.MeanMapKernel` by first doing the kernel
    embedding and then taking the mean.
    Using :class:`sklearn.kernel_approximation.RBFSampler`::

        rbf_mmk = Pipeline([
            ('rbf', BagPreprocesser(RBFSampler(gamma=g, n_components=100))),
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
        Transform a list of bag features into a matrix of its mean features.

        Parameters
        ----------
        X : :class:`skl_groups.features.Features` or list of bag feature arrays
            Data to transform.

        Returns
        -------
        X_new : array, shape ``[len(X), X.dim]``
            X transformed into its means.
        '''
        X = as_features(X)
        return np.vstack([np.mean(bag, axis=0) for bag in X])
