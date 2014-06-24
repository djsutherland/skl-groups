from __future__ import division

import numpy as np
from sklearn.externals.six import iteritems
from sklearn.externals.six.moves import xrange, zip


def _linear(Bs, dim, num_q, rhos, nus, clamp=True):
    # and the rest of the estimator is
    #   B / m * mean(nu ^ -dim)
    est = Bs / num_q * np.mean(nus ** (-dim), axis=0)
    if clamp:
        np.maximum(est, 0, out=est)
    return est


def kl(Ks, dim, num_q, rhos, nus, clamp=True):
    r'''
    Estimate the KL divergence between distributions:
        \int p(x) \log (p(x) / q(x))
    using the kNN-based estimator (5) of
        Qing Wang, Sanjeev R Kulkarni, and Sergio Verdu (2009).
        Divergence Estimation for Multidimensional Densities Via
        k-Nearest-Neighbor Distances.
        IEEE Transactions on Information Theory.
        http://www.ee.princeton.edu/~verdu/reprints/WanKulVer.May2009.pdf
    which is:
        d * 1/n \sum \log (nu_k(i) / rho_k(i)) + log(m / (n - 1))

    If clamp, enforces KL >= 0.

    Returns an array of shape (num_Ks,).
    '''
    est = dim * np.mean(np.log(nus) - np.log(rhos), axis=0)
    est += np.log(num_q / (rhos.shape[0] - 1))
    if clamp:
        np.maximum(est, 0, out=est)
    return est
kl.self_value = 0
kl.needs_alpha = False


def _alpha_div(omas, Bs, dim, num_q, rhos, nus, clamp=True):
    N = rhos.shape[0]

    # the actual main estimate:
    #   rho^(- dim * est alpha) nu^(- dim * est beta)
    #   = (rho / nu) ^ (dim * (1 - alpha))
    # do some reshaping trickery to get broadcasting right
    estimates = (rhos / nus)[:, np.newaxis, :]
    estimates = estimates ** (dim * omas.reshape(1, -1, 1))
    estimates = np.mean(estimates, axis=0)  # shape (n_alphas, n_Ks)

    estimates *= Bs

    # factors based on the sizes:
    #   1 / [ (n-1)^(est alpha) * m^(est beta) ] = ((n-1) / m) ^ (1 - alpha)
    estimates *= ((N - 1) / num_q) ** omas

    if clamp:
        np.maximum(estimates, 0, out=estimates)
    return estimates


def _jensen_shannon_core(Ks, dim, min_i, digamma_vals, num_q, rhos, nus, clamp=True):
    # We need to calculate the mean over points in X of
    # d * log radius of largest ball with no more than M/(n+m-1) weight
    #         where X points have weight 1 / (2 (n-1))
    #           and Y points have weight 1 / (2 m)
    # - digamma(# of neighbors in that ball)

    # ignores clamp.

    # NOTE: this is a stupidly slow implementation. the cython one should
    #       be much better, and also parallelize.
    num_p = rhos.shape[0]

    t = 2 * num_p - 1
    p_wt = 1 / t
    q_wt = num_p / (num_q * t)

    alphas = Ks / (num_p + num_q - 1)

    est = np.zeros(Ks.size)

    max_k = rhos.shape[1]
    combo = np.empty(max_k * 2, dtype=[('dist', np.float32), ('weight', float)])
    # could vectorize this loop if searchsorted worked axis-wise
    for rho, nu, in zip(rhos, nus):
        combo['dist'][:max_k] = rho
        combo['dist'][max_k:] = nu
        combo['weight'][:max_k] = p_wt
        combo['weight'][max_k:] = q_wt
        combo.sort()
        quantiles = np.cumsum(combo['weight'])

        i = quantiles.searchsorted(alphas, side='right')  # number pts in ball
        assert i.min() >= min_i

        est += dim * np.log(combo['dist'][i - 1]) - digamma_vals[i - min_i]
    est /= num_p
    return est


################################################################################

def _estimate_cross_divs(X_features, X_indices, X_rhos,
                         Y_features, Y_indices, Y_rhos,
                         funcs, Ks, max_K, save_all_Ks,
                         n_output, do_sym, to_self,
                         log_progress, n_jobs, min_dist, clamp):
    n_X = len(X_indices)
    n_Y = len(Y_features)

    K_indices = Ks - 1
    which_Ks = slice(None, None) if save_all_Ks else K_indices
    n_Ks = len(Ks)

    outputs = np.empty([n_output, n_Ks, n_X, n_Y, 2 if do_sym else 1],
                       dtype=np.float32)
    outputs.fill(np.nan)

    # Keep track of whether each function needs rho_sub or just rho
    if save_all_Ks:
        def needs_sub(func):
            return not getattr(func, 'needs_all_ks', False)
    else:
        def needs_sub(func):
            return False

    X_features.make_stacked()
    X_feats = X_features.stacked_features
    X_bounds = X_features._boundaries

    # rho: kth neighbors of each X in X
    # nu:  kth neighbors of each X in Y
    # so we can get the nus for all Xs into one Y at once (one column of out)

    Y_indices_loop = log_progress(Y_indices) if log_progress else Y_indices
    for j, Y_index in enumerate(Y_indices_loop):
        # Loop over columns of the output array,
        # searching / getting nus from all of the X bags to this one Y bag.

        # TODO: if X = Y, this wastes time for searches to self

        num_q = Y_features.n_pts[j]

        # KNN search is parallelized with openmp in flann
        knns = np.sqrt(Y_index.nn_index(X_feats, max_K)[1][:, which_Ks])
        np.maximum(min_dist, knns, out=knns)

        # all this is sequential. that's why this implementation is slow.
        for i in xrange(n_X):
            rho = X_rhos[i]
            nu = knns[X_bounds[i]:X_bounds[i + 1]]

            if save_all_Ks:
                rho_sub = rho[:, K_indices]
                nu_sub = nu[:, K_indices]

            for func, info in iteritems(funcs):
                o = (info.pos, slice(None), i, j, 0)

                if i == j and to_self and func.self_value is not None:
                    outputs[o] = func.self_value
                elif needs_sub(func):
                    outputs[o] = func(num_q, rho_sub, nu_sub, clamp=clamp)
                else:
                    outputs[o] = func(num_q, rho, nu, clamp=clamp)

    if do_sym:
        if X_features == Y_features:
            w = slice(None) if do_sym == True else list(do_sym)
            outputs[w, ..., 1] = outputs[w, ..., 0].transpose(0, 1, 3, 2)
            return outputs

        Y_features.make_stacked()
        Y_feats = Y_features.stacked_features
        Y_bounds = Y_features._boundaries

        if do_sym == True:
            sym_funcs = funcs
        else:
            sym_funcs = {func: info for func, info in iteritems(funcs)
                         if not do_sym.isdisjoint(info.pos)}

        X_indices_loop = log_progress(X_indices) if log_progress else X_indices
        for i, X_index in enumerate(X_indices_loop):
            # Loop over rows of the output array,
            # searching / getting nus from all of the Y bags to this one X bag.

            num_q = X_features.n_pts[i]

            knns = np.sqrt(X_index.nn_index(Y_feats, max_K)[1][:, which_Ks])
            np.maximum(min_dist, knns, out=knns)

            for j in xrange(n_Y):
                rho = Y_rhos[j]
                nu = knns[Y_bounds[j]:Y_bounds[j + 1]]

                if save_all_Ks:
                    rho_sub = rho[:, K_indices]
                    nu_sub = nu[:, K_indices]

                for func, info in iteritems(sym_funcs):
                    o = (info.pos, slice(None), i, j, 1)

                    if i == j and to_self and func.self_value is not None:
                        outputs[o] = func.self_value
                    elif needs_sub(func):
                        outputs[o] = func(num_q, rho_sub, nu_sub, clamp=clamp)
                    else:
                        outputs[o] = func(num_q, rho, nu, clamp=clamp)

    return outputs
