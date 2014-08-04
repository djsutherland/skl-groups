from __future__ import division

from collections import namedtuple, defaultdict, OrderedDict
from functools import partial
import itertools
import logging
import warnings

import numpy as np
from scipy.special import gamma, gammaln, psi
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory
from sklearn.externals.six import iteritems, itervalues, string_types
from sklearn.externals.six.moves import map, zip, xrange

try:
    from cyflann import FLANNIndex, FLANNParameters
except ImportError:
    from pyflann import FLANN as FLANNIndex, FLANNParameters

from ..features import as_features
from ..utils import identity, ProgressLogger, as_integer_type
from ._knn import _linear, kl, _alpha_div, _jensen_shannon_core

from ._knn import _estimate_cross_divs
try:
    from skl_groups_accel.knn_divs import (
            _estimate_cross_divs as _estimate_cross_divs_fast)
    have_fast_version = True
except ImportError as e:
    have_fast_version = False
    fast_version_error = e


__all__ = ['KNNDivergenceEstimator']


################################################################################
### Logging setup
logger = logging.getLogger(__name__)

progress_logger = logging.getLogger(__name__ + '.progress')
progress_logger.propagate = False
progress_logger.addHandler(logging.NullHandler())

def plog(it, name=None, total=None):
    return ProgressLogger(progress_logger, name=name)(it, total=total)


################################################################################
### Main class

class KNNDivergenceEstimator(BaseEstimator, TransformerMixin):
    r'''
    Estimates various divergence functions between bags.

    Assumes that each bag represents an independent and identically distributed
    sample from some unknown probability distribution (on which certain
    technical assumptions are made), and estimates various distances between
    them. The valid divergence functions are, where p and q refer to the density
    functions for two bags:

    * 'kl': The `Kullback-Liebler divergence <https://en.wikipedia.org/wiki/KL_divergence>`_,
      which acts like a distance and has some attractive information-theoretic
      properties as well as often giving good results on machine learning
      problems in practice. Values are nonnegative but can go to infinity.
      :math:`\int p(x) \log\left( \frac{p(x)}{q(x)} \right) dx`.
      Estimated as in [1]_.
    * 'renyi:.8' or any other number (other than 1): The
      `Renyi-alpha divergence <https://en.wikipedia.org/wiki/Renyi_divergence#R.C3.A9nyi_divergence>`_,
      :math:`\frac{1}{\alpha - 1} \log \int p(x) \left( \frac{p(x)}{q(x)} \right)^{\alpha - 1} dx`.
      Converges to 'kl' as :math:`\alpha` goes to 1.
      Values are nonnegative, but can go to infinity.
      Estimated as in [2]_.
    * 'tsallis:.8' or any other number (other than 1): The 
      `Tsallis-alpha divergence <https://en.wikipedia.org/wiki/Tsallis_entropy>`_,
      :math:`\frac{1}{\alpha - 1} \left( \int p(x) \left(\frac{p(x)}{q(x)} \right)^{\alpha - 1} dx - 1 \right)`.
      Converges to `kl` as :math:`\alpha` goes to 1.
      Values are nonnegative, but can go to infinity.
      Estimated as in [2]_.
    * 'hellinger': The `Hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_,
      which is a true distance (i.e. it is symmetric and its true value
      satisfies the triangle inequality) and is between 0 and 1.
      Defined by :math:`\sqrt{1 - \int \sqrt{p(x) q(x)} dx}`.
      Estimated as in [2]_.
    * 'bc': The `Bhattacharyya coefficient <https://en.wikipedia.org/wiki/Bhattacharyya_coefficient>`_,
      :math:`\int \sqrt{p(x) q(x)} dx`.
      This is an affinity rather than a distance and is between 0 and 1.
      Estimated as in [2]_.
    * 'l2': The :math:`L_2` distance between density functions
      :math:`\sqrt{ \int (p(x) - q(x))^2 dx }`.
      Estimated as in [2]_.
      A true distance function (symmetric and satisfies the triangle inequality.)
    * 'linear': :math:`\int p(x) q(x) dx`.
      Estimated as in [2]_.
    * 'jensen-shannon' or 'js': The
      `Jensen-Shannon divergence <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_.
      Equal to :math:`\frac{1}{2} D(p \| M) + \tfrac{1}{2} D(q \| M)`,
      where D is the Kullback-Liebler divergence
      and M is an equal mixture between p and q,
      or equivalently :math:`H[M] - \frac{1}{2} H[p] - \frac{1}{2} H[q]`
      where H is the Shannon entropy.
      Symmetric, its square root satisfies the triangle inequality,
      and is between 0 and :math:`\ln 2`.
      Estimated using [3]_.

    Note that even for the true distance functions,
    resulting kernels will not necessarily be positive semidefinite
    due to estimation error.

    Parameters
    ----------

    div_funcs : sequence of strings, optional, default ['kl']
        A sequence of divergence function spec strings, as above.
        For Renyi or Tsallis divergences, you can pass multiple values of
        alpha, e.g. ``['renyi:.9', 'renyi:.8', 'tsallis:.99']``.

    Ks : sequence of positive integers, optional, default [3]
        The Ks to use for the K-nearest-neighbor estimator.
        If you have very small bags (sizes less than, say, 25), try 1 or 2;
        if large, say more than 1000, try 4 or 5.
        Must be less than the smallest bag size; for the proof of [2]_ to work,
        should be at least 3 for most divergence functions.
        Jensen-Shannon treats this parameter a little differently.

    do_sym : boolean, optional, default False
        As well as returning D(X || Y), return D(Y || X).

    n_jobs : integer, optional, default 1
        The number of CPUs to use in the computation. -1 means 'all CPUs'.

    clamp : boolean, optional, default True
        For functions with bounded outputs, "clamp" them to lie within that
        range; for example, if the estimator for KL divergence gives a negative
        value, return 0 instead.

    min_dist : float, optional, default 1e-3
        Protect against nearly-identical points by treating any distances less
        than this amount as this number. Tiny distances screw up the estimators
        because they assume that the inputs are from a continuous distribution,
        where this doesn't happen.

    flann_algorithm : string, optional, default 'auto'
        Which algorithm to use in FLANN for nearest neighbors. Defaults to
        'auto', which chooses 'kdtree_single' if the input dimension is at most
        5 and 'linear' otherwise. In high-ish dimensional spaces, you can get
        much better performance at the cost of approximate answers by using
        other index types; see the FLANN documentation.

    flann_args : dictionary, optional, default {}
        Other arguments to pass to FLANN.

    version : one of {'best', 'fast', 'slow'}, optional, default 'best'
        Whether to use the fast Cython implementation from skl-groups-accel
        or the slower pure-Python implementation. 'best' chooses 'fast' if
        available and you aren't using custom divergence functions, 'slow'
        otherwise.

    memory : Instance of :class:`joblib.Memory` or string, optional
        Used to cache the indices and the output of :meth:`transform`.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    Attributes
    ----------

    `features_` : :class:`skl_groups.features.Features`
        The features passed to `fit` (except made :meth:`sklearn.features.Features.bare`).

    `indices_` : list of :class:`cyflann.FLANNIndex` or :class:`pyflann.FLANN`
        A FLANN index for each bag in `features_`.

    `rhos_` : list of arrays
        For each bag in `features_`, the distance to the Kth nearest neighbor
        of each point amongst its own bag.
        ``rhos_[i][j, k]`` is the distance to either the ``Ks[k]`` th or the
        ``k+1`` th nearest neighbor of ``features_[i][j, :]`` in
        ``features_[i]`` (not including ``features_[i][j, :]`` itself).
        It's the ``k+1`` th if Jensen-Shannon divergence is requested.

        May or may not be present after :meth:`fit`; will be after
        :meth:`transform`.

    See also
    --------
    :class:`skl_groups.kernels.PairwisePicker`
        Chooses a particular divergence function / K combination in a pipeline.
        Doing parameter tuning with :class:`KNNDivergenceEstimator` with a
        :attr:`memory` and then :class:`PairwisePicker` is more efficient
        than directly tuning over :class:`KNNDivergenceEstimator` arguments.


    Notes
    -----
    The convergence proof in [1]_ seems incorrect. The estimator seems to work
    well, in practice, though.

    The Jensen-Shannon estimation is performed by using the estimator of
    [3]_ to get :math:`H[\frac{1}{2}(X + Y)]`, by combining the two samples
    and assigning weights such that the total weight from each sample is equal,
    and subtracting the mean of :math:`H[X]` and :math:`H[Y]` according to the
    same estimator (with equally-weighted points). The ``K`` parameter is
    used for the value of :math:`M = n \alpha` in [3]_.

    References
    ----------
    .. [1] Q. Wang, S. Kulkarni, & S. Verdu (2009).
           Divergence Estimation for Multidimensional Densities Via
           k-Nearest-Neighbor Distances.
           IEEE Transactions on Information Theory, 55(5), 2392-2405.
    .. [2] B. Poczos, L. Xiong, D. J. Sutherland, & J. Schneider (2012).
           Nonparametric kernel estimators for image classification.
           In Computer Vision and Pattern Recognition (CVPR).
    .. [3] H. Hino & N. Murata (2013).
           Information estimators for weighted observations.
           Neural Networks.
    '''
    def __init__(self, div_funcs=('kl',), Ks=(3,), do_sym=False, n_jobs=1,
                 clamp=True, min_dist=1e-3,
                 flann_algorithm='auto', flann_args=None, version='best',
                 memory=Memory(cachedir=None, verbose=0)):
        self.div_funcs = div_funcs
        self.Ks = Ks
        self.do_sym = do_sym
        self.n_jobs = n_jobs
        self.clamp = clamp
        self.min_dist = min_dist
        self.flann_algorithm = flann_algorithm
        self.flann_args = flann_args
        self.version = version
        self.memory = memory

    def _get_Ks(self):
        "Ks as an array and type-checked."
        Ks = as_integer_type(self.Ks)
        if Ks.ndim != 1:
            raise TypeError("Ks should be 1-dim, got shape {}".format(Ks.shape))
        if Ks.min() < 1:
            raise ValueError("Ks should be positive; got {}".format(Ks.min()))
        return Ks

    @property
    def _n_jobs(self):
        "n_jobs, but with the number of cores instead of -1."
        if self.n_jobs == -1:
            from multiprocessing import cpu_count
            return cpu_count()
        return self.n_jobs

    def _flann_args(self, X=None):
        "The dictionary of arguments to give to FLANN."
        args = {'cores': self._n_jobs}
        if self.flann_algorithm == 'auto':
            if X is None or X.dim > 5:
                args['algorithm'] = 'linear'
            else:
                args['algorithm'] = 'kdtree_single'
        else:
            args['algorithm'] = self.flann_algorithm
        if self.flann_args:
            args.update(self.flann_args)

        # check that arguments are correct
        try:
            FLANNParameters().update(args)
        except AttributeError as e:
            msg = "flann_args contains an invalid argument:\n  {}"
            raise TypeError(msg.format(e))

        return args

    def fit(self, X, y=None, get_rhos=False):
        '''
        Sets up for divergence estimation "from" new data "to" X.
        Builds FLANN indices for each bag, and maybe gets within-bag distances.

        Parameters
        ----------
        X : list of arrays or :class:`skl_groups.features.Features`
            The bags to search "to".

        get_rhos : boolean, optional, default False
            Compute within-bag distances :attr:`rhos_`. These are only needed
            for some divergence functions or if do_sym is passed, and they'll
            be computed (and saved) during :meth:`transform` if they're not
            computed here.

            If you're using Jensen-Shannon divergence, a higher max_K may
            be needed once it sees the number of points in the transformed bags,
            so the computation here might be wasted.
        '''
        self.features_ = X = as_features(X, stack=True, bare=True)

        # if we're using a function that needs to pick its K vals itself,
        # then we need to set max_K here. when we transform(), might have to
        # re-do this :|
        Ks = self._get_Ks()
        _, _, _, max_K, save_all_Ks, _ = _choose_funcs(
            self.div_funcs, Ks, X.dim, X.n_pts, None, self.version)

        if max_K >= X.n_pts.min():
            msg = "asked for K = {}, but there's a bag with only {} points"
            raise ValueError(msg.format(max_K, X.n_pts.min()))

        memory = self.memory
        if isinstance(memory, string_types):
            memory = Memory(cachedir=memory, verbose=0)

        self.indices_ = id = memory.cache(_build_indices)(X, self._flann_args())
        if get_rhos:
            self.rhos_ = _get_rhos(X, id, Ks, max_K, save_all_Ks, self.min_dist)
        elif hasattr(self, 'rhos_'):
            del self.rhos_
        return self

    def transform(self, X):
        r'''
        Computes the divergences from X to :attr:`features_`.

        Parameters
        ----------
        X : list of bag feature arrays or :class:`skl_groups.features.Features`
            The bags to search "from".

        Returns
        -------
        divs : array of shape ``[len(div_funcs), len(Ks), len(X), len(features_)] + ([2] if do_sym else [])``
            The divergences from X to :attr:`features_`.
            ``divs[d, k, i, j]`` is the ``div_funcs[d]`` divergence
            from ``X[i]`` to ``fetaures_[j]`` using a K of ``Ks[k]``.
            If ``do_sym``, ``divs[d, k, i, j, 0]`` is
            :math:`D_{d,k}( X_i \| \texttt{features_}_j)` and
            ``divs[d, k, i, j, 1]`` is :math:`D_{d,k}(\texttt{features_}_j \| X_i)`.
        '''
        X = as_features(X, stack=True, bare=True)
        Y = self.features_

        Ks = np.asarray(self.Ks)

        if X.dim != Y.dim:
            msg = "incompatible dimensions: fit with {}, transform with {}"
            raise ValueError(msg.format(Y.dim, X.dim))

        memory = self.memory
        if isinstance(memory, string_types):
            memory = Memory(cachedir=memory, verbose=0)

        # ignore Y_indices to avoid slow pickling of them
        # NOTE: if the indices are approximate, then might not get the same
        #       results!
        est = memory.cache(_est_divs, ignore=['n_jobs', 'Y_indices', 'Y_rhos'])
        output, self.rhos_ = est(
            X, Y, self.indices_, getattr(self, 'rhos_', None),
            self.div_funcs, Ks,
            self.do_sym, self.clamp, self.version, self.min_dist,
            self._flann_args(), self._n_jobs)
        return output


def _choose_funcs(div_funcs, Ks, dim, X_n_pts, Y_n_pts, version):
    funcs_base, metas_base, n_meta_only = _parse_specs(div_funcs, Ks)
    funcs, metas = _set_up_funcs(
        funcs_base, metas_base, Ks, dim, X_n_pts, Y_n_pts)

    max_K = Ks.max()
    for func in funcs:
        if hasattr(func, 'K_needed'):
            max_K = max(max_K, func.K_needed)

    save_all_Ks = any(getattr(f, 'needs_all_ks', False) for f in funcs_base)

    # choose the version we can handle
    non_fastable_funcs = set(funcs_base) - fast_funcs
    if version == 'fast':
        if not have_fast_version:
            msg = "asked for 'fast', but skl_groups_accel not available"
            raise ValueError(msg)
        elif non_fastable_funcs:
            msg = "asked for 'fast', but functions are incompatible: {}"
            raise ValueError(msg.format(non_fastable_funcs))
    elif version == 'slow':
        pass
    elif version == 'best':
        if non_fastable_funcs:
            version = 'slow'
        elif have_fast_version:
            version = 'fast'
        else:
            warnings.warn("Using 'slow' version of KNNDivergenceEstimator, "
                          " because skl_groups_accel isn't available; its "
                          "'fast' version is much faster on large "
                          "problems. Pass version='slow' to suppress this "
                          "warning. \n  {}".format(fast_version_error))
            version = 'slow'
    else:
        msg = "Unknown value '{}' for version."
        raise ValueError(msg.format(version))

    return funcs, metas, n_meta_only, max_K, save_all_Ks, version


def _build_indices(X, flann_args):
    "Builds FLANN indices for each bag."
    # TODO: should probably multithread this
    logger.info("Building indices...")
    indices = [None] * len(X)
    for i, bag in enumerate(plog(X, name="index building")):
        indices[i] = idx = FLANNIndex(**flann_args)
        idx.build_index(bag)
    return indices


def _get_rhos(X, indices, Ks, max_K, save_all_Ks, min_dist):
    "Gets within-bag distances for each bag."
    logger.info("Getting within-bag distances...")

    if max_K >= X.n_pts.min():
        msg = "asked for K = {}, but there's a bag with only {} points"
        raise ValueError(msg.format(max_K, X.n_pts.min()))

    # need to throw away the closest neighbor, which will always be self
    # thus K=1 corresponds to column 1 in the result array
    which_Ks = slice(1, None) if save_all_Ks else Ks

    indices = plog(indices, name="within-bag distances")
    rhos = [None] * len(X)
    for i, (idx, bag) in enumerate(zip(indices, X)):
        r = np.sqrt(idx.nn_index(bag, max_K + 1)[1][:, which_Ks])
        np.maximum(min_dist, r, out=r)
        rhos[i] = r
    return rhos


def _est_divs(X, Y, Y_indices, Y_rhos, div_funcs, Ks,
              do_sym, clamp, version, min_dist, flann_args, n_jobs):

    funcs, metas, n_meta_only, max_K, save_all_Ks, version = _choose_funcs(
        div_funcs, Ks, X.dim, X.n_pts, Y.n_pts, version)

    if not do_sym:
        do_sym = {
            req_pos for f, info in iteritems(metas)
                    for req_pos, req in zip(info.deps, f.needs_results)
                    if req.needs_transpose
        }

    to_self = X == Y
    need_Y_rhos = to_self or do_sym or any(meta.needs_rhos[1] for meta in metas)

    if need_Y_rhos:
        if Y_rhos is not None:
            if save_all_Ks:
                old_max_K = Y_rhos[0].shape[1]
                if max_K > old_max_K:
                    msg = ("Y_rhos had a lower max_K ({}) than we actually "
                           "need ({}); recomputing rhos.")
                    logger.warning(msg.format(old_max_K, max_K))
                    Y_rhos = None
            else:
                if Y_rhos[0].shape[1] != Ks.size:
                    raise ValueError("invalid Y_rhos passed")

        if Y_rhos is None:
            Y_rhos = _get_rhos(Y, Y_indices, Ks, max_K, save_all_Ks, min_dist)

    if to_self:
        X_indices = Y_indices
        X_rhos = Y_rhos
    else:
        X_indices = _build_indices(X, flann_args)
        X_rhos = _get_rhos(X, X_indices, Ks, max_K, save_all_Ks, min_dist)

    logger.info("Getting divergences...")
    if version == 'fast':
        fn = _estimate_cross_divs_fast
    else:
        fn = _estimate_cross_divs
    outputs = fn(X, X_indices, X_rhos, Y, Y_indices, Y_rhos,
                 funcs, Ks, max_K, save_all_Ks, len(div_funcs) + n_meta_only,
                 do_sym, to_self,
                 ProgressLogger(progress_logger, name="cross-divergences"),
                 n_jobs, min_dist, clamp)

    logger.info("Computing meta-divergences...")

    if save_all_Ks:
        X_rhos = [rho[:, Ks - 1] for rho in X_rhos]
        if need_Y_rhos:
            Y_rhos = [rho[:, Ks - 1] for rho in Y_rhos]

    for meta, info in iteritems(metas):
        args = ()
        if meta.needs_rhos[0]:
            args += (X_rhos,)
        if meta.needs_rhos[1]:
            args += (Y_rhos,)
        args += (outputs[info.deps],)

        outputs[info.pos] = meta(*args, clamp=clamp, to_self=to_self)

    if do_sym != True:
        outputs = outputs[:, :, :, :, 0]

    if n_meta_only:
        outputs = np.ascontiguousarray(outputs[:-n_meta_only])

    logger.info("Done with divergences.")
    return outputs, Y_rhos


################################################################################
### Estimators of various divergences based on nearest-neighbor distances.
#
# The standard interface for these functions is:
#
# Function attributes:
#
#   needs_alpha: whether this function needs an alpha parameter. Default false.
#
#   self_value: The value that this function should take when comparing a
#               sample to itself: either a scalar constant or None (the
#               default), in which case the function is still called with
#               rhos = nus.
#
#   chooser_fn: a function that gets alphas (if needs_alpha), Ks, dim, X_ns, Y_ns
#               (the arrays of bag sizes) and returns a partial() of a "core"
#               function, with some things precomputed. If not present, just
#               does partial(fn, [alphas,] Ks, dim).
#
#   needs_all_ks: whether this function needs *all* the neighbor distances up
#                 to the max K value, rather than just the values of K that are
#                 actually used. Default false.
#
#   chooser_fn.returns_ks: whether the chooser_fn returns the max value of K
#                          needed. This allows an estimator function to require
#                          a higher value of K than requested by the user. Only
#                          if needs_all_ks; default false.
#
# Arguments:
#
#   alphas (if needs_alpha; array-like, scalar or 1d): the alpha values to use
#
#   Ks (array-like, scalar or 1d): the K values used
#
#   dim (scalar): the dimension of the feature space
#
#   num_q (scalar): the number of points in the sample from q
#
#   rhos: an array of within-bag nearest neighbor distances for a sample from p.
#         rhos[i, j] should be the distance from the ith sample from p to its
#         Ks[j]'th neighbor in the same sample. Shape: (num_p, num_Ks).
#   nus: an array of nearest neighbor distances from samples from other dists.
#        nus[i, j] should be the distance from the ith sample from p to its
#        Ks[j]'th neighbor in the sample from q. Shape: (num_p, num_Ks).
#
# Returns an array of divergence estimates. If needs_alpha, should be of shape
# (num_alphas, num_Ks); otherwise, of shape (num_Ks,).

def linear(Ks, dim, num_q, rhos, nus):
    r'''
    Estimates the linear inner product \int p q between two distributions,
    based on kNN distances.
    '''
    return _get_linear(Ks, dim)(num_q, rhos, nus)

def _get_linear(Ks, dim, X_ns=None, Y_ns=None):
    # Estimated with alpha=0, beta=1:
    #   B_{k,d,0,1} = (k - 1) / pi^(dim/2) * gamma(dim/2 + 1)
    #   (using gamma(k) / gamma(k - 1) = k - 1)
    Ks = np.reshape(Ks, (-1,))
    Bs = (Ks - 1) / np.pi ** (dim / 2) * gamma(dim / 2 + 1)  # shape (num_Ks,)
    return partial(_linear, Bs, dim)
linear.self_value = None  # have to execute it
linear.needs_alpha = False
linear.chooser_fn = _get_linear

# kl function is entirely in _np_divs (nothing to precompute)

def alpha_div(alphas, Ks, dim, num_q, rhos, nus):
    r'''
    Estimate the alpha divergence between distributions:
        \int p^\alpha q^(1-\alpha)
    based on kNN distances.

    Used in Renyi, Hellinger, Bhattacharyya, Tsallis divergences.

    Enforces that estimates are >= 0.

    Returns divergence estimates with shape (num_alphas, num_Ks).
    '''
    return _get_alpha_div(alphas, Ks, dim)(num_q, rhos, nus)

def _get_alpha_div(alphas, Ks, dim, X_ns=None, Y_ns=None):
    alphas = np.reshape(alphas, (-1, 1))
    Ks = np.reshape(Ks, (1, -1))

    omas = 1 - alphas

    # We're estimating with alpha = alpha-1, beta = 1-alpha.
    # B constant in front:
    #   estimator's alpha = -beta, so volume of unit ball cancels out
    #   and then ratio of gamma functions
    Bs = np.exp(gammaln(Ks) * 2 - gammaln(Ks + omas) - gammaln(Ks - omas))

    return partial(_alpha_div, omas, Bs, dim)

alpha_div.self_value = 1
alpha_div.needs_alpha = True
alpha_div.chooser_fn = _get_alpha_div


def jensen_shannon_core(Ks, dim, num_q, rhos, nus):
    r'''
    Estimates
          1/2 mean_X( d * log radius of largest ball in X+Y around X_i
                                with no more than M/(n+m-1) weight
                                where X points have weight 1 / (2 n - 1)
                                  and Y points have weight n / (m (2 n - 1))
                      - digamma(# of neighbors in that ball))

    This is the core pairwise component of the estimator of Jensen-Shannon
    divergence based on the Hino-Murata weighted information estimator. See
    the docstring for jensen_shannon for an explanation.
    '''
    ns = np.array([rhos.shape[0], num_q])
    return _get_jensen_shannon_core(Ks, dim, ns)[0](num_q, rhos, nus)

def _get_jensen_shannon_core(Ks, dim, X_ns, Y_ns):
    # precompute the max/min possible digamma(i) values: the floors/ceils of
    #
    #   M/(n+m-1) / (1 / (2 n - 1))
    #   M/(n+m-1) / (n / (m (2 n - 1)))
    #
    # for any valid value of n, m.

    min_X_n = np.min(X_ns)
    max_X_n = np.max(X_ns)
    if Y_ns is None:
        min_Y_n = min_X_n
        max_Y_n = max_X_n
    else:
        min_Y_n = np.min(Y_ns)
        max_Y_n = np.max(Y_ns)
    min_K = np.min(Ks)
    max_K = np.max(Ks)

    # figure out the smallest i value we might need (# of neighbors in ball)

    wt_bounds = [np.inf, -np.inf]
    min_wt_n = None; min_wt_m = None
    # max_wt_n = None; max_wt_m = None
    n_ms = list(itertools.product([min_X_n, max_X_n], [min_Y_n, max_Y_n]))
    for n, m in itertools.chain(n_ms, map(reversed, n_ms)):
        base = (2 * n - 1) / (n + m - 1)

        for wt in (base, base * m / n):
            if wt < wt_bounds[0]:
                wt_bounds[0] = wt
                min_wt_n = n
                min_wt_m = m

            if wt > wt_bounds[1]:
                wt_bounds[1] = wt
                # max_wt_n = n
                # max_wt_m = m

    if wt_bounds[0] * min_K < 1:
        msg = "K={} is too small for Jensen-Shannon estimator with n={}, m={}"
        raise ValueError((msg + "; must be at least {}").format(
             min_K, min_wt_n, min_wt_m, int(np.ceil(1 / wt_bounds[0]))))

    min_i = int(np.floor(wt_bounds[0] * min_K))
    max_i = int(np.ceil( wt_bounds[1] * max_K))
    digamma_vals = psi(np.arange(min_i, max_i + 1))

    # TODO: If we don't actually hit the worst case, might be nice to still
    #       run and just nan those elements that we can't compute. This is
    #       over-conservative.
    return partial(_jensen_shannon_core, Ks, dim, min_i, digamma_vals), max_i

jensen_shannon_core.needs_alpha = False
jensen_shannon_core.chooser_fn = _get_jensen_shannon_core
jensen_shannon_core.needs_all_ks = True
jensen_shannon_core.chooser_fn.returns_ks = True
jensen_shannon_core.self_value = np.nan
# The self_value should be the entropy estimate. But since we'll just subtract
# that later, don't bother computing it.


################################################################################
### Meta-estimators: things that need some additional computation on top of
###                  the per-bag stuff of the functions above.

# These functions are run after the base estimators above are complete.
#
# The interface here is:
#
# Function attributes:
#
#   needs_alpha: whether this function needs an alpha parameter. Default false.
#
#   needs_results: a list of MetaRequirement objects (below).
#                  Note that it is legal for meta estimators to depend on other
#                  meta estimators; circular dependencies cause the spec parser
#                  to crash.
#
# Arguments:
#
#   alphas (if needs_alpha; array-like, scalar or 1d): the alpha values to use
#
#   Ks (array-like, scalar or 1d): the K values used
#
#   dim (scalar): the dimension of the feature space
#
#   rhos: a list of within-bag NN distances, each of which is like the rhos
#         argument above
#
#   required: a list of the results array for each MetaRequirement classes,
#             each of shape (num_Ks, n_X, n_Y, 1 or 2),
#             where the last dimension depends on whether we're doing the
#             symmetric or not.
#
# Returns: array of results, shape (num_alphas, num_Ks, n_X, n_Y, 1 or 2).

MetaRequirement = namedtuple('MetaRequirement', 'func alpha needs_transpose')
# func: the function of the regular divergence that's needed
# alpha: None if no alpha is needed. Otherwise, can be a scalar alpha value,
#        or a callable which takes the (scalar or list) alphas for the meta
#        function and returns the required function's alpha(s).
# needs_transpose: if true, ensures the required results have both directions


def bhattacharyya(Ks, dim, required, clamp=True, to_self=False):
    r'''
    Estimate the Bhattacharyya coefficient between distributions, based on kNN
    distances:  \int \sqrt{p q}

    If clamp (the default), enforces 0 <= BC <= 1.

    Returns an array of shape (num_Ks,).
    '''
    est = required
    if clamp:
        est = np.minimum(est, 1)  # BC <= 1
    return est
bhattacharyya.needs_alpha = False
bhattacharyya.needs_rhos = (False, False)
bhattacharyya.needs_results = [MetaRequirement(alpha_div, 0.5, False)]


def hellinger(Ks, dim, required, clamp=True, to_self=False):
    r'''
    Estimate the Hellinger distance between distributions, based on kNN
    distances:  \sqrt{1 - \int \sqrt{p q}}

    Always enforces 0 <= H, to be able to sqrt; if clamp, also enforces
    H <= 1.

    Returns a vector: one element for each K.
    '''
    bc = required
    est = 1 - bc
    np.maximum(est, 0, out=est)
    if clamp:
        np.minimum(est, 1, out=est)
    np.sqrt(est, out=est)
    return est
hellinger.needs_alpha = False
hellinger.needs_rhos = (False, False)
hellinger.needs_results = [MetaRequirement(alpha_div, 0.5, False)]


def renyi(alphas, Ks, dim, required, min_val=np.spacing(1),
          clamp=True, to_self=False):
    r'''
    Estimate the Renyi-alpha divergence between distributions, based on kNN
    distances:  1/(\alpha-1) \log \int p^alpha q^(1-\alpha)

    If the inner integral is less than min_val (default ``np.spacing(1)``),
    uses the log of min_val instead.

    If clamp (the default), enforces that the estimates are nonnegative by
    replacing any negative estimates with 0.

    Returns an array of shape (num_alphas, num_Ks).
    '''
    alphas = np.reshape(alphas, (-1, 1))
    est = required

    est = np.maximum(est, min_val)  # TODO: can we modify in-place?
    np.log(est, out=est)
    est /= alphas - 1
    if clamp:
        np.maximum(est, 0, out=est)
    return est
renyi.needs_alpha = True
renyi.needs_rhos = (False, False)
renyi.needs_results = [MetaRequirement(alpha_div, identity, False)]


def tsallis(alphas, Ks, dim, required, clamp=True, to_self=False):
    r'''
    Estimate the Tsallis-alpha divergence between distributions, based on kNN
    distances:  (\int p^alpha q^(1-\alpha) - 1) / (\alpha - 1)

    If clamp (the default), enforces the estimate is nonnegative.

    Returns an array of shape (num_alphas, num_Ks).
    '''
    alphas = np.reshape(alphas, (-1, 1))
    alpha_est = required

    est = alpha_est - 1
    est /= alphas - 1
    if clamp:
        np.maximum(est, 0, out=est)
    return est
tsallis.needs_alpha = True
tsallis.needs_rhos = (False, False)
tsallis.needs_results = [MetaRequirement(alpha_div, identity, False)]


def l2(Ks, dim, X_rhos, Y_rhos, required, clamp=True, to_self=False):
    r'''
    Estimates the L2 distance between distributions, via
        \int (p - q)^2 = \int p^2 - \int p q - \int q p + \int q^2.

    \int pq and \int qp are estimated with the linear function (in both
    directions), while \int p^2 and \int q^2 are estimated via the quadratic
    function below.

    Always clamps negative estimates of l2^2 to 0, because otherwise the sqrt
    would break.
    '''
    n_X = len(X_rhos)
    n_Y = len(Y_rhos)

    linears = required
    assert linears.shape == (1, Ks.size, n_X, n_Y, 2)

    X_quadratics = np.empty((Ks.size, n_X), dtype=np.float32)
    for i, rho in enumerate(X_rhos):
        X_quadratics[:, i] = quadratic(Ks, dim, rho)

    Y_quadratics = np.empty((Ks.size, n_Y), dtype=np.float32)
    for j, rho in enumerate(Y_rhos):
        Y_quadratics[:, j] = quadratic(Ks, dim, rho)

    est = -linears.sum(axis=4)
    est += X_quadratics[None, :, :, None]
    est += Y_quadratics[None, :, None, :]
    np.maximum(est, 0, out=est)
    np.sqrt(est, out=est)

    # diagonal is of course known to be zero
    if to_self:
        est[:, :, xrange(n_X), xrange(n_Y)] = 0
    return est[:, :, :, :, None]
l2.needs_alpha = False
l2.needs_rhos = (True, True)
l2.needs_results = [MetaRequirement(linear, alpha=None, needs_transpose=True)]


# Not actually a meta-estimator, though it could be if it just repeated the
# values across rows (or columns).
def quadratic(Ks, dim, rhos, required=None):
    r'''
    Estimates \int p^2 based on kNN distances.

    In here because it's used in the l2 distance, above.

    Returns array of shape (num_Ks,).
    '''
    # Estimated with alpha=1, beta=0:
    #   B_{k,d,1,0} is the same as B_{k,d,0,1} in linear()
    # and the full estimator is
    #   B / (n - 1) * mean(rho ^ -dim)
    N = rhos.shape[0]
    Ks = np.asarray(Ks)
    Bs = (Ks - 1) / np.pi ** (dim / 2) * gamma(dim / 2 + 1)  # shape (num_Ks,)
    est = Bs / (N - 1) * np.mean(rhos ** (-dim), axis=0)
    return est


def jensen_shannon(Ks, dim, X_rhos, Y_rhos, required,
                   clamp=True, to_self=False):
    r'''
    Estimate the difference between the Shannon entropy of an equally-weighted
    mixture between X and Y and the mixture of the Shannon entropies:

        JS(X, Y) = H[ (X + Y) / 2 ] - (H[X] + H[Y]) / 2

    We use a special case of the Hino-Murata weighted information estimator with
    a fixed M = n \alpha, about equivalent to the K-nearest-neighbor approach
    used for the other estimators:

        Hideitsu Hino and Noboru Murata (2013).
        Information estimators for weighted observations. Neural Networks.
        http://linkinghub.elsevier.com/retrieve/pii/S0893608013001676


    The estimator for JS(X, Y) is:

        log volume of the unit ball - log M + log(n + m - 1) + digamma(M)
        + 1/2 mean_X( d * log radius of largest ball in X+Y around X_i
                                with no more than M/(n+m-1) weight
                                where X points have weight 1 / (2 n - 1)
                                  and Y points have weight n / (m (2 n - 1))
                      - digamma(# of neighbors in that ball) )
        + 1/2 mean_Y( d * log radius of largest ball in X+Y around Y_i
                                with no more than M/(n+m-1) weight
                                where X points have weight m / (n (2 m - 1))
                                  and Y points have weight 1 / (2 m - 1)
                      - digamma(# of neighbors in that ball) )

        - 1/2 (log volume of the unit ball - log M + log(n - 1) + digamma(M))
        - 1/2 mean_X( d * log radius of the largest ball in X around X_i
                                with no more than M/(n-1) weight
                                where X points have weight 1 / (n - 1))
                      - digamma(# of neighbors in that ball) )

        - 1/2 (log volume of the unit ball - log M + log(m - 1) + digamma(M))
        - 1/2 mean_Y( d * log radius of the largest ball in Y around Y_i
                                with no more than M/(n-1) weight
                                where X points have weight 1 / (m - 1))
                      - digamma(# of neighbors in that ball) )

        =

        log(n + m - 1) + digamma(M)
        + 1/2 mean_X( d * log radius of largest ball in X+Y around X_i
                                with no more than M/(n+m-1) weight
                                where X points have weight 1 / (2 n - 1)
                                  and Y points have weight n / (m (2 n - 1))
                      - digamma(# of neighbors in that ball) )
        + 1/2 mean_Y( d * log radius of largest ball in X+Y around Y_i
                                with no more than M/(n+m-1) weight
                                where X points have weight m / (n (2 m - 1))
                                  and Y points have weight 1 / (2 m - 1)
                      - digamma(# of neighbors in that ball) )
        - 1/2 [log(n-1) + mean_X( d * log rho_M(X_i) )]
        - 1/2 [log(m-1) + mean_Y( d * log rho_M(Y_i) )]
    '''

    X_ns = np.array([rho.shape[0] for rho in X_rhos])
    Y_ns = np.array([rho.shape[0] for rho in Y_rhos])
    n_X = X_ns.size
    n_Y = Y_ns.size

    # cores[0, k, i, j, 0] is mean_X(d * ... - psi(...)) for X[i], Y[j], M=Ks[k]
    # cores[0, k, i, j, 1] is mean_Y(d * ... - psi(...)) for X[i], Y[j], M=Ks[k]
    cores = required
    assert cores.shape == (1, Ks.size, n_X, n_Y, 2)

    # X_bits[k, i] is log(n-1) + mean_X( d * log rho_M(X_i) )  for X[i], M=Ks[k]
    X_bits = np.empty((Ks.size, n_X), dtype=np.float32)
    for i, rho in enumerate(X_rhos):
        X_bits[:, i] = dim * np.mean(np.log(rho), axis=0)
    X_bits += np.log(X_ns - 1)[np.newaxis, :]

    # Y_bits[k, j] is log(n-1) + mean_Y( d * log rho_M(Y_i) )  for Y[j], M=Ks[k]
    Y_bits = np.empty((Ks.size, n_Y), dtype=np.float32)
    for j, rho in enumerate(Y_rhos):
        Y_bits[:, j] = dim * np.mean(np.log(rho), axis=0)
    Y_bits += np.log(Y_ns - 1)[np.newaxis, :]

    est = cores.sum(axis=4)
    est -= X_bits.reshape(1, Ks.size, n_X, 1)
    est -= Y_bits.reshape(1, Ks.size, 1, n_Y)
    est /= 2
    est += np.log(-1 + X_ns[None, None, :, None] + Y_ns[None, None, None, :])
    est += psi(Ks)[None, :, None, None]

    # diagonal is zero
    if to_self:
        est[:, :, xrange(n_X), xrange(n_Y)] = 0

    if clamp:  # know that 0 <= JS <= ln(2)
        np.maximum(0, est, out=est)
        np.minimum(np.log(2), est, out=est)
    return est[:, :, :, :, None]
jensen_shannon.needs_alpha = False
jensen_shannon.needs_rhos = (True, True)
jensen_shannon.needs_results = [
    MetaRequirement(jensen_shannon_core, alpha=None, needs_transpose=True)]


################################################################################
### Parse string specifications into functions to use

func_mapping = {
    'linear': linear,
    'kl': kl,
    'alpha': alpha_div,
    'bc': bhattacharyya,
    'hellinger': hellinger,
    'renyi': renyi,
    'tsallis': tsallis,
    'l2': l2,
    'js-core': jensen_shannon_core,
    'js': jensen_shannon,
    'jensen-shannon': jensen_shannon,
}
fast_funcs = {kl, alpha_div, linear, jensen_shannon_core}


def topological_sort(deps):
    '''
    Topologically sort a DAG, represented by a dict of child => set of parents.
    The dependency dict is destroyed during operation.

    Uses the Kahn algorithm: http://en.wikipedia.org/wiki/Topological_sorting
    Not a particularly good implementation, but we're just running it on tiny
    graphs.
    '''
    order = []
    available = set()

    def _move_available():
        to_delete = []
        for n, parents in iteritems(deps):
            if not parents:
                available.add(n)
                to_delete.append(n)
        for n in to_delete:
            del deps[n]

    _move_available()
    while available:
        n = available.pop()
        order.append(n)
        for parents in itervalues(deps):
            parents.discard(n)
        _move_available()

    if available:
        raise ValueError("dependency cycle found")
    return order


_FuncInfo = namedtuple('_FuncInfo', 'alphas pos')
_MetaFuncInfo = namedtuple('_MetaFuncInfo', 'alphas pos deps')
def _parse_specs(specs, Ks):
    '''
    Set up the different functions we need to call.

    Returns:
        - a dict mapping base estimator functions to _FuncInfo objects.
          If the function needs_alpha, then the alphas attribute is an array
          of alpha values and pos is a corresponding array of indices.
          Otherwise, alphas is None and pos is a list containing a single index.
          Indices are >= 0 if they correspond to something in a spec,
          and negative if they're just used for a meta estimator but not
          directly requested.
        - an OrderedDict mapping functions to _MetaFuncInfo objects.
          alphas and pos are like for _FuncInfo; deps is a list of indices
          which should be passed to the estimator. Note that these might be
          other meta functions; this list is guaranteed to be in an order
          such that all dependencies are resolved before calling that function.
          If no such order is possible, raise ValueError.
        - the number of meta-only results

    # TODO: update doctests for _parse_specs

    >>> _parse_specs(['renyi:.8', 'hellinger', 'renyi:.9'])
    ({<function alpha_div at 0x10954f848>:
            _FuncInfo(alphas=[0.8, 0.5, 0.9], pos=[-1, -2, -3])},
     OrderedDict([
        (<function hellinger at 0x10954fc80>,
            _MetaFuncInfo(alphas=None, pos=[1], deps=[array(-2)])),
        (<function renyi at 0x10954fcf8>,
            _MetaFuncInfo(alphas=[0.8, 0.9], pos=[0, 2], deps=[-1, -3]))
     ]), 3)

    >>> _parse_specs(['renyi:.8', 'hellinger', 'renyi:.9', 'l2'])
    ({<function alpha_div at 0x10954f848>:
        _FuncInfo(alphas=[0.8, 0.5, 0.9], pos=[-1, -2, -3]),
      <function linear at 0x10954f758>: _FuncInfo(alphas=None, pos=[-4])
     }, OrderedDict([
        (<function hellinger at 0x10954fc80>,
            _MetaFuncInfo(alphas=None, pos=[1], deps=[array(-2)])),
        (<function l2 at 0x10954fde8>,
            _MetaFuncInfo(alphas=None, pos=[3], deps=[-4])),
        (<function renyi at 0x10954fcf8>,
            _MetaFuncInfo(alphas=[0.8, 0.9], pos=[0, 2], deps=[-1, -3]))
     ]), 4)

    >>> _parse_specs(['renyi:.8', 'hellinger', 'renyi:.9', 'l2', 'linear'])
    ({<function alpha_div at 0x10954f848>:
        _FuncInfo(alphas=[0.8, 0.5, 0.9], pos=[-1, -2, -3]),
      <function linear at 0x10954f758>: _FuncInfo(alphas=None, pos=[4])
     }, OrderedDict([
        (<function hellinger at 0x10954fc80>,
            _MetaFuncInfo(alphas=None, pos=[1], deps=[array(-2)])),
        (<function l2 at 0x10954fde8>,
            _MetaFuncInfo(alphas=None, pos=[3], deps=[4])),
        (<function renyi at 0x10954fcf8>,
            _MetaFuncInfo(alphas=[0.8, 0.9], pos=[0, 2], deps=[-1, -3]))
     ]), 3)
    '''
    funcs = {}
    metas = {}
    meta_deps = defaultdict(set)

    def add_func(func, alpha=None, pos=None):
        needs_alpha = getattr(func, 'needs_alpha', False)
        is_meta = hasattr(func, 'needs_results')

        d = metas if is_meta else funcs
        if func not in d:
            if needs_alpha:
                args = {'alphas': [alpha], 'pos': [pos]}
            else:
                args = {'alphas': None, 'pos': [pos]}

            if not is_meta:
                d[func] = _FuncInfo(**args)
            else:
                d[func] = _MetaFuncInfo(deps=[], **args)
                for req in func.needs_results:
                    if callable(req.alpha):
                        req_alpha = req.alpha(alpha)
                    else:
                        req_alpha = req.alpha
                    add_func(req.func, alpha=req_alpha)
                    meta_deps[func].add(req.func)
                    meta_deps[req.func]  # make sure required func is in there

        else:
            # already have an entry for the func
            # need to give it this pos, if it's not None
            # and also make sure that the alpha is present
            info = d[func]
            if not needs_alpha:
                if pos is not None:
                    if info.pos != [None]:
                        msg = "{} passed more than once"
                        raise ValueError(msg.format(func_name))

                    info.pos[0] = pos
            else:  # needs alpha
                try:
                    idx = info.alphas.index(alpha)
                except ValueError:
                    # this is a new alpha value we haven't seen yet
                    info.alphas.append(alpha)
                    info.pos.append(pos)
                    if is_meta:
                        for req in func.needs_results:
                            if callable(req.alpha):
                                req_alpha = req.alpha(alpha)
                            else:
                                req_alpha = req.alpha
                            add_func(req.func, alpha=req_alpha)
                else:
                    # repeated alpha value
                    if pos is not None:
                        if info.pos[idx] is not None:
                            msg = "{} with alpha {} passed more than once"
                            raise ValueError(msg.format(func_name, alpha))
                        info.pos[idx] = pos

    # add functions for each spec
    for i, spec in enumerate(specs):
        func_name, alpha = (spec.split(':', 1) + [None])[:2]
        if alpha is not None:
            alpha = float(alpha)

        try:
            func = func_mapping[func_name]
        except KeyError:
            msg = "'{}' is not a known function type"
            raise ValueError(msg.format(func_name))

        needs_alpha = getattr(func, 'needs_alpha', False)
        if needs_alpha and alpha is None:
            msg = "{} needs alpha but not passed in spec '{}'"
            raise ValueError(msg.format(func_name, spec))
        elif not needs_alpha and alpha is not None:
            msg = "{} doesn't need alpha but is passed in spec '{}'"
            raise ValueError(msg.format(func_name, spec))

        add_func(func, alpha, i)

    # number things that are dependencies only
    meta_counter = itertools.count(-1, step=-1)
    for info in itertools.chain(itervalues(funcs), itervalues(metas)):
        for i, pos in enumerate(info.pos):
            if pos is None:
                info.pos[i] = next(meta_counter)

    # fill in the dependencies for metas
    for func, info in iteritems(metas):
        deps = info.deps
        assert deps == []

        for req in func.needs_results:
            f = req.func
            req_info = (metas if hasattr(f, 'needs_results') else funcs)[f]
            if req.alpha is not None:
                if callable(req.alpha):
                    req_alpha = req.alpha(info.alphas)
                else:
                    req_alpha = req.alpha

                find_alpha = np.vectorize(req_info.alphas.index, otypes=[int])
                pos = np.asarray(req_info.pos)[find_alpha(req_alpha)]
                if np.isscalar(pos):
                    deps.append(pos[()])
                else:
                    deps.extend(pos)
            else:
                pos, = req_info.pos
                deps.append(pos)

    # topological sort of metas
    meta_order = topological_sort(meta_deps)
    metas_ordered = OrderedDict(
        (f, metas[f]) for f in meta_order if hasattr(f, 'needs_results'))

    return funcs, metas_ordered, -next(meta_counter) - 1


def _set_up_funcs(funcs, metas_ordered, Ks, dim, X_ns=None, Y_ns=None):
    # replace functions with partials of args
    def replace_func(func, info):
        needs_alpha = getattr(func, 'needs_alpha', False)

        new = None
        args = (Ks, dim)
        if needs_alpha:
            args = (info.alphas,) + args

        if hasattr(func, 'chooser_fn'):
            args += (X_ns, Y_ns)
            if (getattr(func, 'needs_all_ks', False) and
                    getattr(func.chooser_fn, 'returns_ks', False)):
                new, K = func.chooser_fn(*args)
                new.K_needed = K
            else:
                new = func.chooser_fn(*args)
        else:
            new = partial(func, *args)

        for attr in dir(func):
            if not (attr.startswith('__') or attr.startswith('func_')):
                setattr(new, attr, getattr(func, attr))
        return new

    rep_funcs = dict(
        (replace_func(f, info), info) for f, info in iteritems(funcs))
    rep_metas_ordered = OrderedDict(
        (replace_func(f, info), info) for f, info in iteritems(metas_ordered))

    return rep_funcs, rep_metas_ordered
