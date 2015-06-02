from __future__ import division

import operator as op

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.six.moves import xrange, reduce

from ..features import as_features
from ..utils import is_integer

VALID_BASES = frozenset({'cosine'})
class L2DensityTransformer(BaseEstimator, TransformerMixin):
    '''
    Summarizes sets into a single vector, whose inner products estimate the
    L2 inner product between the density functions underlying each of the sets.

    **IMPORTANT**: Assumes that inputs lie in [0, 1] - use
    :class:`skl_groups.preprocessing.BagMinMaxScaler`
    with ``feature_range=(0, 1)``, ``truncate=True``, and optionally a
    ``fit_feature_range`` to ensure this.

    Classifiers built off of this representation can scale to very large set
    sizes (since they're summarized in a fixed-dimensional vector) and/or
    number of sets (since you can do a linear classifier on either them or
    an explicit kernel mapping). But the representation only works for very
    small number of dimensions: the dimensionality of the output is exponential
    in both the smoothness coefficient and the dimension of the input. Some
    specific numbers:

    ==========  =======  =======  =======  =======  =======
    smoothness      d=1      d=2      d=3      d=4      d=5
    ==========  =======  =======  =======  =======  =======
             1        2        3        4        5        6
             2        3        6       11       20       36
             3        4       11       29       70      157
             4        5       17       54      165      482
             5        6       26       99      357    1,203
             6        7       35      163      688    2,673
             7        8       45      239    1,154    5,139
             8        9       58      344    1,867    9,389
             9       10       73      486    2,907   15,930
            10       11       90      648    4,272   25,810
            11       12      106      847    6,070   39,855
            12       13      123    1,069    8,357   59,711
            13       14      146    1,355   11,307   86,346
            14       15      168    1,680   14,998  122,467
            15       16      193    2,046   19,470  168,957
            16       17      216    2,446   24,809  229,743
            17       18      243    2,911   31,211  305,153
            18       19      271    3,443   38,928  400,844
            19       20      302    4,022   47,816  518,154
            20       21      335    4,662   58,201  662,629
    ==========  =======  =======  =======  =======  =======

    For more on the estimator, see [1]_, which effectively uses a pipeline of
    :class:`skl_groups.preprocessing.BagMinMaxScaler`,
    :class:`L2DensityTransformer`,
    :class:`sklearn.kernel_approximation.RBFSampler`,
    and then :class:`sklearn.linear_model.Ridge`.

    Parameters
    ----------
    smoothness : positive int
        The smoothness assumption; lower values mean "smoother", with respect
        to the underlying basis. Higher values make the dimensionality higher.

    basis : 'cosine', optional, default 'cosine'
        The underlying orthonormal function basis to use.
        'cosine', the only current option, uses a Fourier basis of
        cosines with differently-scaled periods.

    do_bounds_check : boolean, optional, default True
        Whether to make sure that the data is in [0, 1] in :meth:`transform`.
        (If you skip this check, and the data isn't in the right scale, you'll
        get silently incorrect results.)
    
    Attributes
    ----------
    `inds_` : integer array of shape [output_dim, input_dim]
        The indices of the basis to use. Output dimension i is the
        product of the value of the ``inds_[i, j]``'th basis function at the
        ``i``'th input coordinate.

    References
    ----------
    .. [1] J. B. Oliva, B. Poczos, & J. Schneider (2014).
           Fast Distribution To Real Regression.
           In Proceedings of the 17th International Conference on
           Artificial Intelligence and Statistics (AISTATS).
    '''

    def __init__(self, smoothness, basis='cosine', do_bounds_check=True):
        self.smoothness = smoothness
        if basis not in VALID_BASES:
            raise ValueError("unknown basis '{}'".format(basis))
        self.basis = basis
        self.do_bounds_check = do_bounds_check

    def _check_fitted(self):
        if not hasattr(self, "inds_"):
            raise AttributeError("Model has not been trained yet.")

    @property
    def dim_(self):
        "The dimension of the inputs, once fitted."
        self._check_fitted()
        return self.inds_.shape[1]

    @property
    def out_dim_(self):
        "The dimension of the output vectors, once fitted."
        self._check_fitted()
        return self.inds_.shape[0]

    def fit(self, X, y=None):
        '''
        Picks the elements of the basis to use for the given data.

        Only depends on the dimension of X. If it's more convenient, you can
        pass a single integer for X, which is the dimension to use.

        Parameters
        ----------
        X : an integer, a :class:`Features` instance, or a list of bag features
            The input data, or just its dimension, since only the dimension is
            needed here.
        '''
        if is_integer(X):
            dim = X
        else:
            X = as_features(X)
            dim = X.dim
        M = self.smoothness

        # figure out the smooth-enough elements of our basis
        inds = np.mgrid[(slice(M + 1),) * dim].reshape(dim, (M + 1) ** dim).T
        self.inds_ = inds[(inds ** 2).sum(axis=1) <= M ** 2]
        return self

    def transform(self, X):
        '''
        Transform a list of bag features into its projection series
        representation.

        Parameters
        ----------
        X : :class:`skl_groups.features.Features` or list of bag feature arrays
            New data to transform. The data should all lie in [0, 1];
            use :class:`skl_groups.preprocessing.BagMinMaxScaler` if not.

        Returns
        -------
        X_new : integer array, shape ``[len(X), dim_]``
            X transformed into the new space.
        '''
        self._check_fitted()
        M = self.smoothness
        dim = self.dim_
        inds = self.inds_
        do_check = self.do_bounds_check

        X = as_features(X)
        if X.dim != dim:
            msg = "model fit for dimension {} but got dim {}"
            raise ValueError(msg.format(dim, X.dim))

        Xt = np.empty((len(X), self.inds_.shape[0]))
        Xt.fill(np.nan)

        if self.basis == 'cosine':  # TODO: put this in a C extension?
            coefs = (np.pi * np.arange(M + 1))[..., :]
            for i, bag in enumerate(X):
                if do_check:
                    if np.min(bag) < 0 or np.max(bag) > 1:
                        raise ValueError("Bag {} not in [0, 1]".format(i))

                # apply each phi func to each dataset point: n x dim x M
                phi = coefs * bag[..., np.newaxis]
                np.cos(phi, out=phi)
                phi[:, :, 1:] *= np.sqrt(2)

                # B is the evaluation of each tensor-prodded basis func
                # at each point: n x inds.shape[0]
                B = reduce(op.mul, (phi[:, i, inds[:, i]] for i in xrange(dim)))

                Xt[i, :] = np.mean(B, axis=0)
        else:
            raise ValueError("unknown basis '{}'".format(self.basis))

        return Xt
