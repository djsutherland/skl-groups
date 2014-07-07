from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals.six.moves import xrange as range

from ..features import as_features


class MeanMapKernel(BaseEstimator, TransformerMixin):
    '''
    Computes a kernel between bags as the mean pairwise evaluation of a kernel
    between the elements of the bags.

    Also known as the maximum mean discrepancy (MMD) when used for a hypothesis
    test for whether two samples come from the same distribution.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, polynomial, exponential chi2 and
        sigmoid kernels. Interpretation of the default value is left to
        the kernel; see :mod:`sklearn.metrics.pairwise`.
        Ignored by other kernels.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    See also
    --------
    :class:`skl_groups.summaries.BagMean` is the equivalent if you have an
    explicit kernel map.
    '''

    _pairwise = False
    _pairwise_output = True

    def __init__(self, kernel="rbf", gamma=None, coef0=1, degree=3,
                 kernel_params=None, n_components=100):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel):
            params['gamma'] = self.gamma
            params['degree'] = self.degree
            params['coef0'] = self.coef0

        return params

    def fit(self, X, y=None):
        '''
        Specify the data to which kernel values should be computed.

        Parameters
        ----------
        X : list of arrays or :class:`skl_groups.features.Features`
            The bags to compute "to".
        '''
        self.features_ = as_features(X, stack=True, bare=True)
        # TODO: could precompute things like squared norms if kernel == "rbf".
        # Probably should add support to sklearn instead of hacking it here.
        return self

    def transform(self, X):
        '''
        Compute kernels from X to :attr:`features_`.

        Parameters
        ----------
        X : list of arrays or :class:`skl_groups.features.Features`
            The bags to compute "from". Must have same dimension as
            :attr:`features_`.

        Returns
        -------
        K : array of shape ``[len(X), len(features_)]``
            The kernel evaluations from X to :attr:`features_`.
        '''

        X = as_features(X, stack=True, bare=True)
        Y = self.features_

        if X.dim != Y.dim:
            raise ValueError("MMK transform got dimension {} but had {} at fit"
                             .format(X.dim, Y.dim))

        pointwise = pairwise_kernels(X.stacked_features, Y.stacked_features,
                                     metric=self.kernel,
                                     filter_params=True,
                                     **self._get_kernel_params())

        # TODO: is there a way to do this without a Python loop?
        K = np.empty((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = pointwise[X._boundaries[i]:X._boundaries[i+1],
                                    Y._boundaries[j]:Y._boundaries[j+1]].mean()

        return K
